"""Read-only Blender audit of externally referenced, material-reachable textures."""
import argparse
import json
import os
from pathlib import Path
import sys
import bpy

def reachable(tree,start_nodes,seen=None):
    seen=set() if seen is None else seen; result=[]
    for node in start_nodes:
        key=(tree.as_pointer(),node.as_pointer())
        if key in seen: continue
        seen.add(key); result.append(node)
        if node.type=='GROUP' and node.node_tree:
            outputs=[n for n in node.node_tree.nodes if n.type=='GROUP_OUTPUT']
            result+=reachable(node.node_tree,outputs,seen)
        for socket in node.inputs:
            result+=reachable(tree,[link.from_node for link in socket.links],seen)
    return result

def inspect(path):
    bpy.ops.wm.open_mainfile(filepath=path,load_ui=False,use_scripts=False)
    used={}; color_images=set()
    materials={slot.material for obj in bpy.data.objects if obj.type in {'MESH','CURVES','CURVE'} for slot in obj.material_slots if slot.material}
    for material in materials:
        if not material.use_nodes or not material.node_tree: continue
        outputs=[n for n in material.node_tree.nodes if n.type=='OUTPUT_MATERIAL' and n.is_active_output]
        nodes=reachable(material.node_tree,outputs)
        for node in nodes:
            if node.type=='TEX_IMAGE' and node.image:
                used.setdefault(node.image.name,{'image':node.image,'materials':set()})['materials'].add(material.name)
            socket_name='Base Color' if node.type=='BSDF_PRINCIPLED' else 'Color' if node.type in {'BSDF_DIFFUSE','EMISSION'} else None
            if socket_name and node.inputs.get(socket_name):
                for parent in reachable(material.node_tree,[link.from_node for link in node.inputs[socket_name].links]):
                    if parent.type=='TEX_IMAGE' and parent.image: color_images.add(parent.image.name)
    records=[]
    for name,item in sorted(used.items()):
        im=item['image']
        if im.source not in {'FILE','TILED'}: continue
        path=bpy.path.abspath(im.filepath,library=im.library)
        tiles=list(im.tiles) if im.source=='TILED' else []
        packed_count=len(im.packed_files)
        packed=bool(im.packed_file) if not tiles else packed_count>=len(tiles)
        paths=[]
        for tile in tiles or [None]:
            resolved=path
            if tile is not None:
                resolved=resolved.replace('<UDIM>',str(tile.number)).replace('<UVTILE>',f'u{(tile.number-1001)%10+1}_v{(tile.number-1001)//10+1}')
            paths.append({'tile':tile.number if tile else None,'path':os.path.normpath(resolved),'exists':os.path.isfile(resolved)})
        missing=not packed and any(not p['exists'] for p in paths)
        records.append({'name':name,'source':im.source,'packed':packed,'paths':paths,'missing':missing,
          'color_input':name in color_images,'materials':sorted(item['materials'])})
    return {'images':records,'missing_images':[r for r in records if r['missing']],
      'missing_color_images':[r for r in records if r['missing'] and r['color_input']],
      'material_count':len(materials)}

p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--rank',type=int,required=True); p.add_argument('--world',type=int,required=True)
a=p.parse_args(sys.argv[sys.argv.index('--')+1:]); source=Path(a.input); payload=json.loads(source.read_text())
out=source.parent/f'cache_rank{a.rank}.jsonl'
if out.exists(): raise FileExistsError(out)
for row in payload['caches'][a.rank::a.world]:
    try: record=dict(row,**inspect(row['path']),audit_error=None)
    except Exception as exc: record=dict(row,audit_error=str(exc))
    with out.open('a') as f: f.write(json.dumps(record,ensure_ascii=False)+'\n')
    print('TEXTURE_AUDIT '+json.dumps({'key':row['cache_key'],'error':record['audit_error'],
      'missing':len(record.get('missing_images',[])),'missing_color':len(record.get('missing_color_images',[]))}),flush=True)
(source.parent/f'RANK_{a.rank}_COMPLETE').write_text('success\n')

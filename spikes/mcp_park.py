import json,sys,time
from pathlib import Path
m=json.loads(Path(sys.argv[1]).read_text()); d=Path(sys.argv[2])
for line in sys.stdin:
    row=json.loads(line); meth=row.get('method'); r={}
    if meth=='initialize': r={'protocolVersion':'2024-11-05','capabilities':{'tools':{}},'serverInfo':{'name':'hermes','version':'1'}}
    elif meth=='tools/list': r={'tools':m}
    elif meth=='tools/call':
        (d/'call.json').write_text(line)
        while not (d/'go').exists(): time.sleep(.2)
        r={'content':[{'type':'text','text':(d/'go').read_text()}]}
    if 'id' in row: print(json.dumps({'jsonrpc':'2.0','id':row['id'],'result':r}),flush=True)

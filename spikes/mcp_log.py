import json,sys
from pathlib import Path
m=json.loads(Path(sys.argv[1]).read_text())
log=open(sys.argv[2],'a')
for line in sys.stdin:
    row=json.loads(line); log.write(line); log.flush()
    meth=row.get('method'); r={}
    if meth=='initialize': r={'protocolVersion':'2024-11-05','capabilities':{'tools':{}},'serverInfo':{'name':'hermes','version':'1'}}
    elif meth=='tools/list': r={'tools':m}
    elif meth=='tools/call': r={'content':[{'type':'text','text':'Sunny, 21C'}]}
    if 'id' in row: print(json.dumps({'jsonrpc':'2.0','id':row['id'],'result':r}),flush=True)

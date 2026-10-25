// Runs only in an opaque-origin sandbox iframe, with network restricted to the Pyodide CDN.
const ready=(async()=>{importScripts('https://cdn.jsdelivr.net/pyodide/v0.28.3/full/pyodide.js');const py=await loadPyodide({indexURL:'https://cdn.jsdelivr.net/pyodide/v0.28.3/full/'});await py.loadPackage(['numpy','pandas','matplotlib']);return py;})();
ready.then(()=>self.postMessage({runtimeReady:true})).catch(error=>self.postMessage({id:0,error:'Python加载失败：'+String(error?.stack||error)}));
self.onmessage=async e=>{try{const py=await ready;py.FS.writeFile('/home/pyodide/data.json',e.data.data);try{py.FS.unlink('/home/pyodide/final.png');}catch{}py.globals.set('_source',e.data.code);const value=await py.runPythonAsync(`
import ast, json, base64, io, contextlib, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
os.chdir('/home/pyodide')
plt.close('all')
tree = ast.parse(_source)
allowed = {'numpy','pandas','matplotlib','json','math','statistics'}
for n in ast.walk(tree):
    if isinstance(n, ast.Import):
        if any(a.name.split('.')[0] not in allowed for a in n.names):
            raise ValueError('Only plotting and numeric imports are supported')
    if isinstance(n, ast.ImportFrom) and (n.level or (n.module or '').split('.')[0] not in allowed):
        raise ValueError('Unsupported import')
    if isinstance(n, ast.Attribute) and n.attr.startswith('_'):
        raise ValueError('Private attributes are not supported')
    if isinstance(n, ast.Name) and n.id.startswith('__'):
        raise ValueError('Private names are not supported')
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
        if n.func.id in {'eval','exec','compile','getattr','setattr','globals','locals','vars','input','breakpoint'}:
            raise ValueError('Unsupported dynamic operation')
        if n.func.id == 'open' and (not n.args or not isinstance(n.args[0], ast.Constant) or n.args[0].value != 'data.json' or (len(n.args)>1 and (not isinstance(n.args[1],ast.Constant) or n.args[1].value not in {'r','rt'}))):
            raise ValueError('Read only data.json with open()')
log = io.StringIO()
with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
    exec(compile(tree, 'generated.py', 'exec'), {'__name__':'__main__'})
with Image.open('final.png') as img:
    if img.width > 2000 or img.height > 1600:
        raise ValueError('Image must be at most 2000 x 1600')
    img.verify()
with open('final.png', 'rb') as f:
    raw = f.read()
if len(raw) > 1700000:
    raise ValueError('PNG must be smaller than 1.7 MB')
json.dumps({'image':'data:image/png;base64,'+base64.b64encode(raw).decode(), 'log':log.getvalue()[-3000:]})
`);self.postMessage({id:e.data.id,...JSON.parse(value)});}catch(error){self.postMessage({id:e.data.id,error:String(error.message||error).slice(-5000)});}};

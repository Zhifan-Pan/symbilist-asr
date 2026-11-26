"""Replace exactly the three numbered result tables in the authoritative manuscript."""
from pathlib import Path
import re
import subprocess
import sys
import tempfile
BASE=Path(__file__).resolve().parents[1]
TEX=BASE/'manuscript/main.tex'
def main():
    source=TEX.read_text()
    pattern=re.compile(r'\\begin\{table(\*?)\}.*?\\end\{table\1\}',re.S)
    meta=BASE/'docs/table_iii_sources.json'
    with tempfile.TemporaryDirectory() as tmp:
        out=Path(tmp)/'table.tex'
        for name, script in [('main','table_iii.py'),('stage2','render_table_iv.py'),('ablation','render_table_v.py')]:
            args=['--inputs' if name=='main' else '--results',str(BASE/'results'),'--output',str(out)]
            if name=='main':
                args+=['--provenance',str(meta)]
            subprocess.run([sys.executable,str(BASE/'scripts'/script),*args],check=True)
            matches=[m for m in pattern.finditer(source) if r'\label{tab:'+name+'}' in m[0]]
            assert len(matches)==1, name
            m=matches[0]; source=source[:m.start()]+out.read_text().rstrip()+source[m.end():]
    TEX.write_text(source)
if __name__=='__main__':main()

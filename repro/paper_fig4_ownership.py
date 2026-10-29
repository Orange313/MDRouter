"""Rebuild ownership-sensitivity Figure4 from frozen lifecycle CSV; no new timing."""
import argparse,csv,hashlib,json,math
from pathlib import Path
SCENARIOS=[('OCR_EXCLUSIVE_NEW_INDEX','black',0),('OCR_SHARED_HALF_NEW_INDEX','blue',2),('OCR_PREPAID_NEW_INDEX','red',2)]
def render(rows):
    fmt=lambda x:format(x,'.2f')
    put=lambda x,y,body:r'\put('+fmt(x)+','+fmt(y)+'){'+body+'}'
    box=lambda body:r'\makebox(0,0){\scriptsize '+body+'}'
    line=lambda dx,dy,n:r'\line('+str(dx)+','+str(dy)+'){'+fmt(n)+'}'
    out=[r'\begin{figure*}[t]\centering',r'\setlength{\unitlength}{1pt}',r'\begin{picture}(476,137)'];values=[]
    for model,left in [('ColQwen2',30),('ColPali',270)]:
        width=190;bottom=27;height=89;lo=.75;hi=3.5
        xx=lambda n:left+(math.log10(n)-2)*width/3
        yy=lambda ratio:bottom+(ratio-lo)*height/(hi-lo)
        out += [put(left,bottom,line(1,0,width)),put(left,bottom,line(0,1,height)),put(left+width/2,128,box(model+' -- R100/FullV total cost'))]
        for n,label in [(100,'100'),(1000,'1k'),(10000,'10k'),(100000,'100k')]:out.append(put(xx(n),18,box(label)))
        for ratio in [1,2,3]:out.append(put(left-10,yy(ratio),box(str(ratio))))
        out.append(put(left+width/2,6,box('Conditional query volume $N$')))
        out.append(rf'\qbezier[35]({left},{yy(1):.2f})({left+width/2:.2f},{yy(1):.2f})({left+width},{yy(1):.2f})')
        for scenario,color,dots in SCENARIOS:
            a=next(x for x in rows if x['model']==model and x['scenario']==scenario and int(x['N'])==100)
            fs=float(a['FullV_setup_s']);rs=float(a['R100_setup_s']);fw=(float(a['FullV_total_wall_s'])-fs)/100;rw=(float(a['R100_total_wall_s'])-rs)/100
            ns=[10**(2+3*i/24) for i in range(25)];pts=[];out.append(r'\begingroup\color{'+color+'}')
            for n in ns:
                ratio=(rs+n*rw)/(fs+n*fw);pts.append((xx(n),yy(ratio)));values.append({'model':model,'scenario':scenario,'N_accounting_grid':n,'R100_over_FullV':ratio,'scope':'component projection, not new requests'})
            for (x1,y1),(x2,y2) in zip(pts,pts[1:]):out.append(r'\qbezier'+('['+str(dots)+']' if dots else '')+f'({x1:.2f},{y1:.2f})({(x1+x2)/2:.2f},{(y1+y2)/2:.2f})({x2:.2f},{y2:.2f})')
            if scenario=='OCR_EXCLUSIVE_NEW_INDEX':
                for a in [x for x in rows if x['model']==model and x['scenario']==scenario]:
                    x=xx(int(a['N']));yl=yy(float(a['ratio_warm_only_95_low']));yh=yy(float(a['ratio_warm_only_95_high']));out += [put(x,yl,line(0,1,yh-yl)),put(x-2,yl,line(1,0,4)),put(x-2,yh,line(1,0,4))]
            out.append(r'\endgroup')
    out += [r'\end{picture}',r'\caption{Ownership-sensitive R100/FullV component wall-cost ratios from the new009 cohort. Black: exclusive OCR; blue dotted: 50\% OCR allocation; red dotted: prepaid OCR, each with a new visual index. Connecting curves are accounting projections, not measured request volumes. Black bars at the four prescribed volumes cover warm-timing variation only; singleton setup uncertainty is excluded. No cross-model latency superiority is implied.}\label{fig:ownership}\end{figure*}']
    return '\n'.join(out)+'\n',values
def main():
    p=argparse.ArgumentParser();p.add_argument('--lifecycle-csv',required=True);p.add_argument('--output',required=True);a=p.parse_args();source=Path(a.lifecycle_csv);rows=list(csv.DictReader(source.open(encoding='utf-8-sig')));out=Path(a.output);out.mkdir(parents=True,exist_ok=False);tex,values=render(rows);(out/'paper_figure4.tex').write_text(tex,encoding='utf8',newline='\n')
    with (out/'figure4_accounting_points.csv').open('w',encoding='utf8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    (out/'figure4_receipt.json').write_text(json.dumps({'input_SHA256':hashlib.sha256(source.read_bytes()).hexdigest(),'TeX_SHA256':hashlib.sha256((out/'paper_figure4.tex').read_bytes()).hexdigest(),'GPU':0,'new_timing_requests':0,'scientific_values_changed':False},indent=2)+'\n')
if __name__=='__main__':main()


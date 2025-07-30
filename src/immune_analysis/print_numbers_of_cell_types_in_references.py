import rpy2.robjects as ro
from rpy2.robjects.packages import importr

xCell2 = importr("xCell2")
ro.r('data(PanCancer.xCell2Ref,   package="xCell2")')
ro.r('data(TMECompendium.xCell2Ref, package="xCell2")')

get_sigs = ro.r('xCell2::getSignatures')
names    = ro.r['names']

pan_base = { str(n).split("#")[0] for n in names(get_sigs(ro.r("PanCancer.xCell2Ref"))) }
tme_base = { str(n).split("#")[0] for n in names(get_sigs(ro.r("TMECompendium.xCell2Ref"))) }

print(len(pan_base), len(tme_base))   # → 25 25

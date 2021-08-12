import re
fn={}
with open("zlog.txt","r+") as f:
    for l in f.readlines():
        if "ZQ_backup: fn->name=" in l:
            fname=re.findall(r"ZQ_backup: fn->name=(.+)",l)[0]
            if fname in fn.keys():
                fn[fname] += 1
            else:
                fn[fname]=1
with open("fname.csv","w") as f:
    for key in fn:
        f.write(f"{key},{fn[key]}\n")
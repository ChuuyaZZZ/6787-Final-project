from pathlib import Path
import random
import os
import shutil

pathlist1 = Path("/Users/lixiaoyu/6787-Final-project/full_data/train/fifty").glob('**/*.jpg')
pathlist2 = Path("/Users/lixiaoyu/6787-Final-project/full_data/train/five").glob('**/*.jpg')
pathlist3 = Path("/Users/lixiaoyu/6787-Final-project/full_data/train/fivehundred").glob('**/*.jpg')
pathlist4 = Path("/Users/lixiaoyu/6787-Final-project/full_data/train/hundred").glob('**/*.jpg')
pathlist5 = Path("/Users/lixiaoyu/6787-Final-project/full_data/train/ten").glob('**/*.jpg')
pathlist6 = Path("/Users/lixiaoyu/6787-Final-project/full_data/train/thousand").glob('**/*.jpg')
pathlist7 = Path("/Users/lixiaoyu/6787-Final-project/full_data/train/twenty").glob('**/*.jpg')

pathlist8 = Path("/Users/lixiaoyu/6787-Final-project/full_data/valid/fifty").glob('**/*.jpg')
pathlist9 = Path("/Users/lixiaoyu/6787-Final-project/full_data/valid/five").glob('**/*.jpg')
pathlist10 = Path("/Users/lixiaoyu/6787-Final-project/full_data/valid/fivehundred").glob('**/*.jpg')
pathlist11 = Path("/Users/lixiaoyu/6787-Final-project/full_data/valid/hundred").glob('**/*.jpg')
pathlist12 = Path("/Users/lixiaoyu/6787-Final-project/full_data/valid/ten").glob('**/*.jpg')
pathlist13 = Path("/Users/lixiaoyu/6787-Final-project/full_data/valid/thousand").glob('**/*.jpg')
pathlist14 = Path("/Users/lixiaoyu/6787-Final-project/full_data/valid/twenty").glob('**/*.jpg')

bigpathlist = [pathlist1,pathlist2,pathlist3,pathlist4,pathlist5,pathlist6,pathlist7,pathlist8,pathlist9,pathlist10,pathlist11,pathlist12,pathlist13,pathlist14]


nof_samples = 500
idx = 1
for pathlist in bigpathlist:
    rc = []
    for k, path in enumerate(pathlist):
        if k < nof_samples:
            rc.append(str(path)) # because path is object not string
        else:
            i = random.randint(0, k)
            if i < nof_samples:
                rc[i] = str(path)
    print(pathlist)
    print(len(rc))
    print(idx)
    #print(rc)
    if idx == 1:
        for k in rc:
            shutil.move(k,'/Users/lixiaoyu/6787-Final-project/500_data/train/fifty')
    elif idx == 2:
        for k in rc:
            shutil.move(k,'/Users/lixiaoyu/6787-Final-project/500_data/train/five')
    elif idx == 3:
        for k in rc:
            shutil.move(k,'/Users/lixiaoyu/6787-Final-project/500_data/train/fivehundred')
    elif idx == 4:
        for k in rc:
            shutil.move(k,'/Users/lixiaoyu/6787-Final-project/500_data/train/hundred')
    elif idx == 5:
        for k in rc:
            shutil.move(k,'/Users/lixiaoyu/6787-Final-project/500_data/train/ten')
    elif idx == 6:
        for k in rc:
            shutil.move(k,'/Users/lixiaoyu/6787-Final-project/500_data/train/thousand')
    elif idx == 7:
        for k in rc:
            shutil.move(k,'/Users/lixiaoyu/6787-Final-project/500_data/train/twenty')
    elif idx == 8:
        for k in rc:
            shutil.move(k,'/Users/lixiaoyu/6787-Final-project/500_data/valid/fifty')
    elif idx == 9:
        for k in rc:
            shutil.move(k,'/Users/lixiaoyu/6787-Final-project/500_data/valid/five')
    elif idx == 10:
        for k in rc:
            shutil.move(k,'/Users/lixiaoyu/6787-Final-project/500_data/valid/fivehundred')
    elif idx == 11:
        for k in rc:
            shutil.move(k,'/Users/lixiaoyu/6787-Final-project/500_data/valid/hundred')
    elif idx == 12:
        for k in rc:
            shutil.move(k,'/Users/lixiaoyu/6787-Final-project/500_data/valid/ten')
    elif idx == 13:
        for k in rc:
            shutil.move(k,'/Users/lixiaoyu/6787-Final-project/500_data/valid/thousand')
    elif idx == 14:
        for k in rc:
            shutil.move(k,'/Users/lixiaoyu/6787-Final-project/500_data/valid/twenty')
    idx += 1
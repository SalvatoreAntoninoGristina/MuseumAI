import sys 
import random
import csv
import json
import numpy as np
import detectron2
import os, json, cv2, random
import math
import pandas
import itertools
import pickle
from tqdm import tqdm
from PIL import Image
from PIL import Image, ImageOps, ImageDraw
from os import listdir
from os.path import isfile, join
from datetime import datetime
from pathlib import Path
from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode
from detectron2.config import get_cfg

def create_initial_cfgs():
    
    with open('originali/output_dati/cfgs_dict.pkl', 'rb') as f:
        cfgs_dict = pickle.load(f)
    return cfgs_dict

def o2t(x, y, bx, by, bw,  bh,    # dati del punto e del box di origine
              tx, ty, tbw, tbh):  # dati del box di arrivo
    """
    la box di origine ha l'angolo sinistro in alto in (bx, by) ed e' di dimensioni (bw, bh)
    La box di arrivo ha l'angolo sinistro in alto in (tx, ty) ed e' di dimensioni (tbw, tbh)
     
    trasporta le coordinate x, y allinterno della box di origine
    nelle coordinate out_x, out_y all'interno della box di arrivo

    """
    sx = float(x-bx)/float(bw)
    sy = float(y-by)/float(bh)

    tsx = sx * float(tbw)
    tsy = sy * float(tbh)

    out_x = tx + int(tsx)
    out_y = ty + int(tsy)

    return out_x, out_y

def bbox_from_mask(immagine_mask):
    m_obj = immagine_mask
    # calcolo la bounding box della maschera
    om_data = np.asarray(m_obj)
    res = np.where(om_data == np.amax(om_data))

    #print("\nRES",res)
    om_c_max = np.amax(res[1])
    om_c_min = np.amin(res[1])
    #print("\nDESTRA SI OTTINE DA np.amax res[1]",om_c_max)
    om_r_max = np.amax(res[0])
    om_r_min = np.amin(res[0])

    #print(om_c_max, om_c_min, om_r_max, om_r_min)

    # sinistra alto destra basso
    return om_c_min, om_r_min, om_c_max, om_r_max 

def background_and_object(immagine_oggetto, maschera_oggetto, immagine_background, punto, frazione):
    tx = punto[0]
    ty = punto[1]
    fra = frazione
    c_max2 = immagine_background.size[0]
    bkg = immagine_background
    obj = immagine_oggetto
    
    
    m_obj = maschera_oggetto
    # calcolo la bounding box della maschera
    om_data = np.asarray(m_obj)

    #print("\nGRANDEZZA MASCHERA (ALTEZZA, LARGHEZZA) om_data: ", om_data.shape)
    # l'apsect ration della bounding box 
    sinistra, alto, destra,  basso = bbox_from_mask(maschera_oggetto)
    #print("\nSINISTRA", sinistra)
    #print("\nALTO", alto)
    #print("\nDESTRA", destra)
    #print("\nBASSO", basso)
    # om_ar = (om_c_max-om_c_min)/(om_r_max-om_r_min)
    om_ar = (destra - sinistra) / (basso - alto)
    #print("\n (destra - sinistra) / (basso - alto) = om_ar = ", om_ar)
    # calcolo la bounding box di arrivo sulla immagine bkg
    # width su c, x
    tw = int(c_max2 * fra)
    # height su r, y
    th = round(tw/om_ar)

    #print("\ncalcolo la bounding box di arrivo sulla immagine bkg:\n width su c, x\ntw = int(c_max2 * fra) = ",tw)
    #print("\n height su r, y \nth = round(tw/om_ar) = ",th)

    m_bkg = np.zeros((bkg.size[1], bkg.size[0]))
    #print("m_bkg = np.zeros((bkg.size[1], bkg.size[0])) = ",m_bkg)
    #img = Image.fromarray(m_bkg)

    risultato = bkg.copy()

    #print("\n\nPUNTO X tx: ",tx)
    #print("\n\nPUNTO tx+tw = : ",tx+tw)
    #print("\nPUNTO Y ty: ",ty)
    #print("\nPUNTO  ty+th = : ",ty+th)
    #print("\nLARGHEZZA BACKGROUND c_max2: ",c_max2)
    for x in range(tx, tx + tw):
        
            for y in range(ty, ty + th):
               
                    if x < m_bkg.shape[1]  and y < m_bkg.shape[0] :

                        # trasformo in coordinate sulla maschera dell'oggetto
                        m_obj_x , m_obj_y = o2t(x,y, tx, ty, tw, th, 
                                                    sinistra, alto, 
                                                    destra - sinistra, basso - alto )
                        if m_obj_x < om_data.shape[1] and m_obj_y <  om_data.shape[0] : 
                            
                            if om_data[m_obj_y][m_obj_x] > 0:             
                                m_bkg[y][x] = 255
                                #print("\ny: ",y)
                                #print("\nx: ",x)
                                risultato.putpixel((x,y), obj.getpixel((m_obj_x, m_obj_y)))
                                
                            else:
                                m_bkg[y][x] = 0
                
        

    return risultato, m_bkg

def is_edge(msk,i,j):
    if i == msk.shape[0]-1 or j == msk.shape[1]-1:
        return True
    if msk[i-1,j-1]==0 or msk[i,j-1]==0 or msk[i-1,j]==0 or msk[i+1,j+1]==0 or msk[i,j+1]==0 or msk[i+1,j]==0:
        return True
    else:
        return False

def cart2pol(x, y):
    r = np.sqrt(x**2 + y**2)
    t = np.arctan2(y, x)
    return(r, t)

def pol2cart(r, t):
    x = r * np.cos(t)
    y = r * np.sin(t)
    return(x, y)

def sort_pixel(list_tuple_edge):
    #center = random.choice(list_tuple_edge)
    x_min = min(list_tuple_edge, key = lambda t:t[0])
    x_max = max(list_tuple_edge, key = lambda t:t[0])
    print(x_min[0])
    x_mean = round((x_max[0]+x_min[0])/2)
    y_min = min(list_tuple_edge, key = lambda t:t[1])
    y_max = max(list_tuple_edge, key = lambda t:t[1])
    y_mean = round((y_max[1]+y_min[1])/2)
    center = (x_mean,y_mean)
    print("CENTRO: ",center)
    polar = []
    X=[]
    Y=[]
    for t in list_tuple_edge:

        x = t[0]-center[0]
        y = t[1]-center[1]
        
        #print(x)
        radius , theta = cart2pol(x,y) 
        polar.append((radius,theta))
    
    polar = sorted(polar, key=lambda tup: tup[1], reverse=True)
    polar_to_xy=[]
    for t in polar:
        x , y = pol2cart(t[0],t[1])
        x = round(x)
        y = round(y)
        polar_to_xy.append((x+center[0],y+center[1]))
    
    for t in polar_to_xy:
        X.append(t[0])
        Y.append(t[1])
    
    return X,Y

def all_points_segmentation(msk):
    x=[]
    y=[]
    edge = []
    
    for i in range(0,msk.shape[0]):
        for j in range(0,msk.shape[1]):
            if msk[i,j]==255:
                if is_edge(msk,i,j): 
                    edge.append((j,i))
        
    x,y = sort_pixel(edge[0::5])
    
    return x, y,edge

def add_model(model_name, n_img):
    #to do creare il modello e addestrarlo
    BASEPATH_OGGETTI = "originali/oggetti"
    BASEPATH_SFONDI = "originali/background"
    BASEPATH_OUT = "originali/output_dati"
    # lista di directory da usare per gli oggetti
    l_oggetti = [model_name]
    N = n_img
    # lista delle directory per gli sfondi
    l_sfondi = ["airport_inside", "artstudio", "auditorium", "bakery", 
                "bar", "bathroom", "bedroom", "bookstore", "bowling", 
                "buffet", "casino", "children_room", "church_inside", 
                "classroom", "cloister", "closet", "clothingstore", 
                "computerroom", "concert_hall", "corridor", "deli", "dentaloffice", 
                "dining_room", "elevator", "fastfood_restaurant", "florist", 
                "gameroom", "garage", "greenhouse", "grocerystore", "gym",
                "hairsalon", "hospitalroom", "inside_bus", "inside_subway", 
                "jewelleryshop", "kindergarden", "kitchen", "laboratorywet", 
                "laundromat", "library", "livingroom", "lobby", "locker_room", 
                "mall", "meeting_room", "movietheater", "museum", "nursery", 
                "office", "operating_room", "pantry", "poolinside", 
                "prisoncell", "restaurant", "restaurant_kitchen", "shoeshop",
                "stairscase", "studiomusic", "subway", "toystore", "trainstation", 
                "tv_studio", "videostore", "waitingroom", "warehouse", "winecellar"]
    num_progressivo = 5051
    for obj in l_oggetti:
        
            # fissa la directory degli oggetti
            path = os.path.join(BASEPATH_OGGETTI, obj)

            # fissa la dir di output
            out_path = os.path.join(BASEPATH_OUT, obj)

            # ricava l'elenco delle immagini, non le maschere
            onlyfiles = [f for f in listdir(path) if isfile(join(path, f)) and ("_mask_" not in f) ]

            for i in tqdm (range(N)):
                try:
                    # recupera una immagine a caso con la maschera
                    ff = random.choice(onlyfiles)

                    # costruisce il nome della maschera
                    filename, file_extension = os.path.splitext(ff)
                    ff_mask = filename+"_mask_"+file_extension

                    # prende uno sfondo a caso
                    bckg_dir = os.path.join(BASEPATH_SFONDI, random.choice(l_sfondi) )
                    background = random.choice( [f for f in listdir(bckg_dir)  ] )
                    bb = os.path.join(bckg_dir, background)

                    # apre le immagini
                    i_obj = Image.open(os.path.join(path, ff) )
                    i_mask = Image.open(os.path.join(path,ff_mask)).convert("L")
                    i_bkg = Image.open(bb)

                    # SE LA IMMAGINE HA UN TAG EXIF ROTATED ALLORA 
                    # RUOTA LA IMMAGINE 
                    i_obj_new = ImageOps.exif_transpose(i_obj)

                    # INVERTE LA IMMAGINE DELLA MASCHERA
                    # la maschera deve essere bianca su fondo nero, ma sono disegnate 
                    # nere su fondo biano
                    if(filename not in ["IMG_0001","IMG_0002"]):
                        i_mask = ImageOps.invert(i_mask)

                    # stampa le varie dimensioni 
                    # DEBUG ==============================================
                    #print("immagine: ", ff)
                    #print("dimensioni immagine oggetto", i_obj_new.size)
                    #print("dimensioni maschera oggetto", i_mask.size)
                    #print("img oggetto ", i_obj_new.width, i_obj_new.height)
                    #print("img maschera", i_mask.width, i_mask.height)
                    # =================================================
                    

                    frac = random.uniform(0.25, 0.75)
                    k = 0.3

                    # decide il puntoi  di ancora
                    c_max = i_bkg.size[0] - int(i_bkg.size[0] * k)
                    r_max = i_bkg.size[1] - int(i_bkg.size[1] * k)
                
                    punto = ( random.randint(0, c_max), random.randint(0, r_max ) )

                    print("\nnome file: ",filename)

                    # sovrappone le immagini
                    ris, msk = background_and_object(i_obj_new, i_mask, i_bkg, punto, frac)

                    # ricavo la bounding box della maschera
                    bbox = bbox_from_mask(msk)

                    # DEBUG ==================================================
                    # sovrappone la bounding box per la visualizzazione
                    #temp = ImageDraw.Draw(ris)
                    #temp.rectangle(bbox, outline ="red")
                    #temp.show()
                    # =======================================================

                    #ris.show()

                    # genera il nome di output
                    num_progressivo += 1
                    ID =  filename + "_out_" + str(num_progressivo) + file_extension
                    ff_out = os.path.join( out_path, ID)

                    # salva la immagine generata
                    ris.save(ff_out)

                    # crea la riga relativa alla immagine
                    LICENZA = " "
                    FILENAME = ID
                    IM_HEIGHT = int(ris.height)
                    IM_WIDTH = int(ris.width)
                    DATA = datetime.now()
                    FLICKR_URL = " "
                    CATEGORIA =  obj
                    BBOX = ( bbox[0], bbox[1], bbox[2], bbox[3] )
                    AREA = str( ( bbox[2] - bbox[0] ) * ( bbox[3] - bbox[1] ) )
                    ALL_X, ALL_Y, EDGE = all_points_segmentation(msk)
                    
                    
                    fieldnames = ['LICENZA','FILENAME', 'IM_HEIGHT','IM_WIDTH', 'DATA', 'FLICKR_URL', 'CATEGORIA', 'BBOX','AREA', 'ALL_X_POINTS', 'ALL_Y_POINTS']
                    with open('originali/output_dati/fileout_temp_segmentation2.0.csv', 'a') as myfile:
                        writer = csv.DictWriter(myfile, fieldnames=fieldnames)
                        writer.writerow({'LICENZA': LICENZA, 'FILENAME': FILENAME, 'IM_HEIGHT': IM_HEIGHT, 'IM_WIDTH': IM_WIDTH, 
                        'DATA': DATA, 'FLICKR_URL': FLICKR_URL, 'CATEGORIA': CATEGORIA, 'BBOX': BBOX, 'AREA': AREA, 'ALL_X_POINTS': ALL_X,'ALL_Y_POINTS': ALL_Y })
                except Exception: 
                    continue
    
    with open('originali/output_dati/cfgs_dict.pkl', 'rb') as f:
        cfgs_dict = pickle.load(f)

    cfgs_dict['cfg_'+model_name]='modello_'+model_name+'/config.yml'

    with open('originali/output_dati/cfgs_dict.pkl', 'wb') as f:
        pickle.dump(cfgs_dict, f, pickle.HIGHEST_PROTOCOL)
    train_model(model_name)

def train_model(model_name):
    create_metadata("cfg_"+model_name,"3.0")

    cfg2 = get_cfg()
    cfg2.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg2.DATASETS.TRAIN = (model_name+"_seg"+"3.0",)
    cfg2.DATASETS.TEST = ()
    cfg2.DATALOADER.NUM_WORKERS = 2
    cfg2.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg2.SOLVER.IMS_PER_BATCH = 2
    cfg2.SOLVER.BASE_LR = 0.00025  
    cfg2.SOLVER.MAX_ITER = 1000    
    cfg2.SOLVER.STEPS = []        
    cfg2.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   
    cfg2.MODEL.ROI_HEADS.NUM_CLASSES = 1  
    cfg2.OUTPUT_DIR = 'modello_'+model_name
    os.makedirs(cfg2.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg2) 
    trainer.resume_or_load(resume=True)
    trainer.train()

    f = open(cfg2.OUTPUT_DIR+'/config.yml', 'w')
    f.write(cfg2.dump())
    f.close()
    
def remove_model(model_name):

    with open('originali/output_dati/cfgs_dict.pkl', 'rb') as f:
        cfgs_dict = pickle.load(f)

    del cfgs_dict['cfg_'+model_name]

    with open('originali/output_dati/cfgs_dict.pkl', 'wb') as f:
        pickle.dump(cfgs_dict, f, pickle.HIGHEST_PROTOCOL)

def create_df(name_model):
    df = pandas.read_csv('originali/output_dati/fileout_temp_segmentation2.0.csv')
    model_df = df[(df['CATEGORIA'] == name_model)]
    model_df.reset_index(inplace=True)

    return model_df

def get_dict(img_dir,df):
    dataset_dicts = []

    for a in range(0,df.shape[0]-1):
        record = {}
        objs = []
        record["file_name"] = os.path.join(img_dir,df['FILENAME'][a])
        record["height"] = df['IM_HEIGHT'][a]
        record["width"] = df['IM_WIDTH'][a]
        px = [int(i) for i in df['ALL_X_POINTS'][a].strip('][').split(', ')]
        py = [int(i) for i in df['ALL_Y_POINTS'][a].strip('][').split(', ')]
        #print(px)
        poly = [(x + 0.5, y +0.5 ) for x, y in zip(px, py)]
        poly = [p for x in poly for p in x]
        obj = {
            "bbox": [int (s) for s in df['BBOX'][a].replace("(","").replace(")","").split(',')] ,
            "bbox_mode": BoxMode.XYXY_ABS,
            "segmentation": [poly],
            "category_id": 0,
        }
        objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts

def create_metadata(cfg,ver):
    name_model = cfg[4:]
    d = 'originali/output_dati/'+name_model
    df = create_df(name_model)
    try:
        DatasetCatalog.register(name_model+"_seg"+ver , lambda d=d: get_dict(d,df))
    except Exception:
        pass
    MetadataCatalog.get(name_model+"_seg"+ver).set(thing_classes=[name_model])
    metadata = MetadataCatalog.get(name_model+"_seg"+ver)

    return metadata

def load_cfgs(cfgs_dict,ver):
    loaded_cfgs = {}
    for cfg in cfgs_dict.keys():
        path = cfgs_dict[cfg]
        cfg_to_load = get_cfg()
        cfg_to_load.merge_from_file(path)
        #cfg_to_load.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.95 
        predictor = DefaultPredictor(cfg_to_load)
        meta_data = create_metadata(cfg,ver)
        loaded_cfgs[cfg] = [predictor,0.0,meta_data]
        
    return loaded_cfgs

def find_model_max_score(score_dct):
    v=list(score_dct.values())
    j = itertools.count(0)
    max_score,key_number = max([(i[1], next(j)) for i in v ], key=lambda item:item[0])

    return key_number

def visualizer(dct,output,key,im):
    v = Visualizer(im[:, :, ::-1],
                metadata=dct[key][2], 
                scale=0.2, 
                instance_mode=ColorMode.IMAGE_BW   
    )
    out = v.draw_instance_predictions(output["instances"].to("cpu"))
    #cv2.imshow("a",out.get_image()[:, :, ::-1])
    #cv2.waitKey(0) 
    #cv2.destroyAllWindows()
    cv2.imwrite("destination.jpg",out.get_image()[:, :, ::-1])

def feed_forward(img, modelli_caricati):
    outputs=[]
    im = cv2.imread(img)
    for model in modelli_caricati.keys():
        outputs.append(modelli_caricati[model][0](im))
        try:
            modelli_caricati[model][1] = outputs[-1]['instances'].get('scores').max().item()
        except:
            pass
        
    print(modelli_caricati)
    nth_key_max_score = find_model_max_score(modelli_caricati)
    key_model= list(modelli_caricati.keys())[nth_key_max_score]
    output = outputs[nth_key_max_score]
    visualizer(modelli_caricati, output,key_model,im)
    
    return modelli_caricati

def load_image_into_numpy_array(data):
    npimg = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

if __name__ == '__main__':
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    """come aggiungere un modello, addestrarlo e visualizzare come ha lavorato 
    #add_model("mattarella",1000)
    #feed_forward( "originali/oggetti/mattarella/20210507_152848.JPG",load_cfgs(create_initial_cfgs(),'3.0'))
    dicts = get_dict('originali/output_dati/mattarella', create_df("mattarella"))
    for b in random.sample(dicts, 4):
        img = cv2.imread(b["file_name"])
        
        visualizer = Visualizer(img[:, :, ::-1], metadata=create_metadata("mattarella","3.0"), scale=1)
        out = visualizer.draw_dataset_dict(b)
        cv2.imshow("",out.get_image()[:, :, ::-1])
        cv2.waitKey(0) 
        cv2.destroyAllWindows()
   """
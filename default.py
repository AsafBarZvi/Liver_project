import json
from easydict import EasyDict as edict

default_config = {
    "gpu"               : None,                         #gpu device used (set to None to take the first available device)
    "data_dir"          : "data",                       #data dir containing train.txt test.txt
    "batch_size"        : 15,                           #batch size
    "weight_decay"      : 0.000005,                     #decay on weights
    "bias_decay"        : 0.0000005,                    #decay on biases
    "snapdir"           : "./snaps",                    #path to save snapshots
    "epochs"            : 30,                           #number of epoches
    "logdir"            : "logs",                       #tensorboard log dir
    "summary_interval"  : 200,                          #num of interval to dump summary
    "val_interval"      : 500,                          #num of interval to dump summary
    "lr_values"         : "0.0001;0.00001;0.0000001",   #lr step values
    "lr_boundaries"     : "25;28",                      #epoches to jump between lr values
    "momentum"          : 0.9,                          #momentum
    "continue_training" : False,                        #resume training from latest checkpoint
    "checkpoint_file"   : None,                         #resume from specific ckpt file
    "num_workers"       : 3,                            #data feed pipeline workers
    "max_snapshots_keep": 10,                           #max snaps to keep 0 means all
    "output_dir"        : "infer_out"                   #detections images output path

}

config =  edict(default_config)

def printCfg():
    print "CFG: {"
    for (k,v) in config.__dict__.iteritems():
        print "\t\t\"{}\" : {},".format(k , v if type(v) != str else "\"{}\"".format(v))
    print "}"



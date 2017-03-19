#ifndef YOLO_TRAIN_H
#define YOLO_TRAIN_H
#include "network.h"
#include "region_layer.h"
#include "detection_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "demo.h"
#include "option_darknet_list.h"


/** @brief Train YOLO v1 network */
void train_yolo_v1(char *cfgfile, char *weightfile, char* c_fl_train, char* backup_directory);

/** @brief Train YOLO v2 network */
void train_yolo_v2(char *cfgfile, char *weightfile, char* c_fl_train, char* backup_directory);


#endif

#include "yolo_test.h"
#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#endif

/***************************************************************************************/
void test_yolo_v1(network* net, char *filename, char** names, image** alphabet, char* output_file, float thresh, int visualize)
{
	int j;
	float nms=.4;

	// Allocate memory
	detection_layer l = net->layers[net->n-1];
	box *boxes = calloc(l.side*l.side*l.n, sizeof(box));
	float **probs = calloc(l.side*l.side*l.n, sizeof(float *));
	for(j = 0; j < l.side*l.side*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));

	// Load and resize image
	image im = load_image_color(filename,0,0);
	image sized = resize_image(im, net->w, net->h);
	float *X = sized.data;

	// Predict
	network_predict(*net, X);

	// Get boxes in image coordinates
	get_detection_boxes(l, 1, 1, thresh, probs, boxes, 0);

	// Non maximal suppression
	if (nms) do_nms_sort(boxes, probs, l.side*l.side*l.n, l.classes, nms);

#ifdef OPENCV
	if(visualize){ 
		draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, names, alphabet, l.classes);
		show_image(im, "predictions");
		cvWaitKey(1);
	}
#endif

	// Write to disk
	FILE *fout_box    = fopen(output_file, "w");
        int ibox;
        int numbox = l.side*l.side*l.n;
	
	int instance = -1; // Not a groundtruth
	for(ibox = 0; ibox < numbox; ++ibox)
	{
		int i_class = max_index(probs[ibox], l.classes);
		float prob = probs[ibox][i_class];
		if ( prob < thresh )
			continue;
		box b = boxes[ibox];

		int left  = (b.x-b.w/2.)*im.w;
		int right = (b.x+b.w/2.)*im.w;
		int top   = (b.y-b.h/2.)*im.h;
		int bot   = (b.y+b.h/2.)*im.h;

		if(left < 0) left = 0;
		if(right > im.w-1) right = im.w-1;
		if(top < 0) top = 0;
		if(bot > im.h-1) bot = im.h-1;

        	fprintf(fout_box, "%d %d ", instance, l.classes);
		int class_i = 0;
		for(class_i = 0; class_i < l.classes; class_i++){
			prob = probs[ibox][class_i];
			fprintf(fout_box, " %d %f", class_i,  prob);
		}
		
		int num_vertices = 4; // four corners
                fprintf(fout_box, " %d %d %d %d %d %d %d %d %d", num_vertices, left, top, right, top, right, bot, left, bot);
		fprintf(fout_box, "\n");

	}
	fclose(fout_box);

	free_image(im);
	free_image(sized);
}


/***************************************************************************************/
void test_yolo_v2(network* net, char *filename, char** names, image** alphabet, char* output_file, float thresh, int visualize)
{
        int j;
        float nms=.4;
	float hier_thresh=0.5;

        // Allocate memory
        layer l = net->layers[net->n-1];
        box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
        float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
        for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(l.classes + 1, sizeof(float *));

        // Load and resize image
        image im = load_image_color(filename,0,0);
        image sized = resize_image(im, net->w, net->h);
        float *X = sized.data;

        // Predict
        network_predict(*net, X);

        // Get boxes in image coordinates
        get_region_boxes(l, 1, 1, thresh, probs, boxes, 0, 0, hier_thresh);

        // Non maximal suppression
        if (l.softmax_tree && nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        else if (nms) do_nms_sort(boxes, probs, l.w*l.h*l.n, l.classes, nms);



#ifdef OPENCV
	if(visualize){
		draw_detections(im, l.w*l.h*l.n, thresh, boxes, probs, names, alphabet, l.classes);
		show_image(im, "predictions");
		cvWaitKey(1);
	}
#endif

        // Write to disk
        FILE *fout_box    = fopen(output_file, "w");
        int ibox;
        int numbox = l.w*l.h*l.n;

        int instance = -1; // Not a groundtruth
        for(ibox = 0; ibox < numbox; ++ibox)
        {
                int i_class = max_index(probs[ibox], l.classes);
                float prob = probs[ibox][i_class];
                if ( prob < thresh )
                        continue;
                box b = boxes[ibox];

                int left  = (b.x-b.w/2.)*im.w;
                int right = (b.x+b.w/2.)*im.w;
                int top   = (b.y-b.h/2.)*im.h;
                int bot   = (b.y+b.h/2.)*im.h;

                if(left < 0) left = 0;
                if(right > im.w-1) right = im.w-1;
                if(top < 0) top = 0;
                if(bot > im.h-1) bot = im.h-1;

        	fprintf(fout_box, "%d %d ", instance, l.classes);
                int class_i = 0;
                for(class_i = 0; class_i < l.classes; class_i++){
                        prob = probs[ibox][class_i];
                        fprintf(fout_box, " %d %f", class_i,  prob);
                }

                int num_vertices = 4; // four corners
                fprintf(fout_box, " %d %d %d %d %d %d %d %d %d\n", num_vertices, left, top, right, top, right, bot, left, bot);
                printf("[YOLO] %d %d %d %d %d %d\n", ibox, num_vertices, left, top, right, bot);

        }
        fclose(fout_box);

        free_image(im);
        free_image(sized);
        free(boxes);
        free_ptrs((void **)probs, l.w*l.h*l.n);
}


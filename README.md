# Change_Background_Kmeans
Changing the Green background of an image using the K means method.

**Features**:</br>
* Making use of percentage of dominant color.
* Working better than some other color spaces like hsv and hence no need to convert to other color spaces.
* Allows you to give different number of clusters and different image sizes for different folders.
* Allows you to give multiple folders (contained in one parent folder) or just one folder or just one image to be converted with desired background image.

**Steps to reproduce:**</br>
* You can run this notebook on an image or a folder containing all the images to have desired background other than green.</br>
* The parser here takes four options with two of them as mandatory:
    * 'fore_dir' -> source of images to be converted. You can give path to just a single image or a directory. It will work out on its own.
    * 'bg_dir'   -> source of background images. You can give path to just a single image or a directory. It will work out on its own.
    * '-n'       -> Number of clusters for K means to decide effectively for removing the green background. The default value is set at 6.
    * '-i'       -> The image size, default is 640.
  
* **Execution**:
     * python ch_bg_kmeans.py --fore_dir <path_to_original_images_or_image> --bg_dir <path_to_the_new_background_images_or_image> -n <number_of_clusters> -i <image_size>

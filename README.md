# Change_Background_Kmeans
Changing the Green background of an image using the K means method.

Steps to reproduce:</br>
You can run this notebook on an image or a folder containing all the images to have desired background other than green.</br>
The parser here takes four options with two of them as mandatory: </br>
  'fore_dir' -> source of images to be converted. You can give path to just a single image or a directory. It will work out on its own.</br>
  'bg_dir'   -> source of background images. You can give path to just a single image or a directory. It will work out on its own.</br>
  '-n'       -> Number of clusters for K means to decide effectively for removing the green background. The default value is set at 6.</br>
  '-i'       -> The image size, default is 640.</br>
  
  Execution:</br>
  python ch_bg_kmeans.py --fore_dir <path_to_original_images_or_image> --bg_dir <path_to_the_new_background_images_or_image> -n <number_of_clusters> -i <image_size>

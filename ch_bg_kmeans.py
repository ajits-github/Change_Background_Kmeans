import matplotlib.pyplot as plt
import numpy as np
import cv2
import os, random
import shutil
import argparse
import time
from sklearn.cluster import KMeans

# %matplotlib inline


############## Enable the lines below the comments suffixed with '# 1' to see the patch of dominant colors and print them too for an image  ##############

class Change_BG:
    def __init__(self, opt) -> None:
        self.fore_dir = opt.fore_dir
        self.bg_dir = opt.bg_dir
        self.n_clusters = opt.n_clusters
        self.img_size = opt.img_size
        self.total_images = 0
        self.result_dir = ''

    def main(self):
        print('working..')
        if os.path.isdir(self.fore_dir):
            for dir in os.listdir(self.fore_dir):
                image_path = os.path.join(self.fore_dir, dir)

                if os.path.isdir(image_path): # if multiple folders are being converted
                    self.result_dir = os.path.join(os.path.dirname(self.fore_dir), 'converted_images', dir) # new result dir for converted images
                    if os.path.exists(self.result_dir):
                        shutil.rmtree(self.result_dir)
                    os.makedirs(self.result_dir)
                    for img in os.listdir(image_path):
                        image_path1 = os.path.join(image_path, img)
                        self.change_bg(image_path1)

                else: # if only one folder is being converted
                    self.result_dir = os.path.join(os.path.dirname(self.fore_dir), 'converted_images', os.path.basename(self.fore_dir))
                    if not os.path.exists(self.result_dir):
                        os.makedirs(self.result_dir)
                    self.change_bg(image_path)

            print(f'Converted images are saved at: {os.path.dirname(self.result_dir)}')

        else: # if only one image is being converted
            self.result_dir = os.path.join(os.path.dirname(os.path.dirname(self.fore_dir)), 'converted_images')
            if os.path.exists(self.result_dir):
                    shutil.rmtree(self.result_dir)
            os.makedirs(self.result_dir)
            self.change_bg(self.fore_dir) 
            print(f'Converted images are saved at: {self.result_dir}')
        
        # return self.total_images


    def visualize_colors(self, cluster, centroids):
        # Get the number of different clusters, create histogram, and normalize
        labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
        (hist, _) = np.histogram(cluster.labels_, bins = labels)
        hist = hist.astype("float")
        hist /= hist.sum()

        # Create frequency rect and iterate through each cluster's color and percentage # 1
        # rect = np.zeros((50, 300, 3), dtype=np.uint8) 
        # start = 0

        colors = sorted([(percent, color) for (percent, color) in zip(hist, centroids)])
        color_dict = {}
        green_code = []

        # for (percent, color) in colors: # 1
        for (_, color) in colors:

            # Enable to print the percentage distribution of the colors for an image # 1
            # print(color, "{:0.2f}%".format(percent * 100))

            colors_list = list(map(int, color))
            color_dict.update({float(color[1]) : tuple(colors_list)})
            green_code.append(color[1])
            
            # Enable to see the patch of dominant colors for an image # 1
            # end = start + (percent * 300)
            # cv2.rectangle(rect, (int(start), 0), (int(end), 50), \
            #               color.astype("uint8").tolist(), -1)
            # start = end

        # return rect, color_dict, green_code # 1
        return color_dict, green_code

    def set_green_threshold(self, image):
    # Load image and convert to a list of pixels
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size))
        reshape = image.reshape((image.shape[0] * image.shape[1], 3))

        # Find and display most dominant colors
        cluster = KMeans(self.n_clusters).fit(reshape)

        # visualize, color_dict, green_code = visualize_colors(cluster, cluster.cluster_centers_) # 1
        color_dict, green_code = self.visualize_colors(cluster, cluster.cluster_centers_)

        val_Ugreen = list(color_dict.values())[-1]
        U_GREEN = [i+25 if i+25<=255 else 255 for i in val_Ugreen]
        green_code = [i for i in green_code if len(str(int(i))) > 2]
        green_code.sort()

        if green_code:
            val_lgreen = list(color_dict.__getitem__(green_code[0]))
            L_GREEN = [i-40 if i-40>=0 else 0 for i in val_lgreen]
        else:
            L_GREEN = [50, 80, 40]

        # print(green_code)
        # print(color_dict)
        # print(L_GREEN)

        # Enable it to visualize the patch of all colors in the image # 1
        # visualize = cv2.cvtColor(visualize, cv2.COLOR_BGR2RGB)
        # cv2.imshow('visualize', visualize) 
        # cv2.waitKey()

        return L_GREEN, U_GREEN

    def change_bg(self, image_path):
        
        assert os.path.isdir(self.result_dir), f'The directory for the converted files to be saved has not been created: {self.result_dir}'
        
        # Enable this to know which image you are failing in case it fails
        # print('Working on image:', os.path.relpath(image_path))
           
        image = cv2.imread(image_path)
        image_copy = np.copy(image)
        image_copy1 = np.copy(image)

        L_GREEN, U_GREEN = self.set_green_threshold(image_copy1)

        assert L_GREEN < U_GREEN, f"Lower green '{L_GREEN}' threshold is higher than Upper green '{U_GREEN}' threshold. Check if the background is non-green."

        image_copy = cv2.resize(image_copy, (self.img_size, self.img_size))
        image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

        mask = cv2.inRange(image_copy, np.array(L_GREEN), np.array(U_GREEN))
        plt.imshow(mask, cmap='gray')

        masked_image = np.copy(image_copy)
        masked_image[mask != 0] = [0, 0, 0]
        # masked_image = cv2.cvtColor(masked_image, cv2.COLOR_HSV2RGB) # covert hsv to rgb 

        if os.path.isdir(self.bg_dir): # check if bg image is a dir or image
            bg_img = random.choice(os.listdir(self.bg_dir))
            bg_img = os.path.join(self.bg_dir, bg_img)
        else:
            bg_img = self.bg_dir

        bg_img = cv2.imread(bg_img)
        bg_img = cv2.resize(bg_img, (self.img_size, self.img_size))
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
        crop_background = bg_img[0:self.img_size, 0:self.img_size]

        crop_background[mask == 0] = [0,0,0]

        complete_image = masked_image + crop_background
        complete_image = cv2.cvtColor(complete_image, cv2.COLOR_BGR2RGB)
        complete_image = cv2.resize(complete_image, (self.img_size, self.img_size))

        new_img_path = os.path.join(self.result_dir , os.path.basename(image_path))
        cv2.imwrite(new_img_path, complete_image)
        self.total_images += 1
        

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('fore_dir', type=str, help='source of images to be converted')
    parser.add_argument('bg_dir', type=str, help='source of background images')
    parser.add_argument('-n', '--n-clusters', type=int, default=6, help='initial number of clusters')
    parser.add_argument('-i', '--img-size', type=int, default=640, help='image size')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt




if __name__ == '__main__':
    t0 = time.time()
    opt = parse_opt(True)
    print(opt)
    ch_bg = Change_BG(opt)
    ch_bg.main()
    duration = f'{(time.time() - t0):.3f}'
    print(f"Avg time taken per image: {float(duration) if ch_bg.total_images == 1 else float(ch_bg.total_images) / float(duration)}sec")





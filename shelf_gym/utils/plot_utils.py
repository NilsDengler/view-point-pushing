import numpy as np
import matplotlib.pyplot as plt
import cv2

class Plot():
    def __init__(self, sim, p_id):
        self.sim = sim
        self._p = p_id
        #self.plt_fig = plt.figure()
        self.plt_obj = []
        self.plt_axs = None
        self.first_time = True

    def get_bin_index_from_int(self, index, col_dim):
        row = index / col_dim
        col = index % col_dim
        return int(row), int(col)

    def plt_img(self, imgs, row_dim, col_dim, save=False, pic_save_path=None):
        fig, axs = plt.subplots(row_dim, col_dim)
        for idx, im in enumerate(imgs):
            r, c = self.get_bin_index_from_int(idx, col_dim)
            axs[r, c].imshow(im)
        if save:  plt.savefig(pic_save_path + "rec_sample_images_" + layer + "_epoch_" + str(epoch + 1))

    def plt_multi_layer(self, imgs, row_dim, col_dim):
        fig, axs = plt.subplots(row_dim, col_dim)
        fig2, axs2 = plt.subplots(row_dim, col_dim)
        for idx, im in enumerate(imgs):
            r, c = self.get_bin_index_from_int(idx, col_dim)
            axs[r, c].imshow(im[:, :, 0])
            axs2[r, c].imshow(im[:, :, 1])


    def plt_single_img_multi_layer(self, img, save=False, path=None, name=None):
        fig, axs = plt.subplots(2)
        axs[0].imshow(img[:, :, 0])
        axs[1].imshow(img[:, :, 1])
        if save:
            plt.savefig( path+name)
            #print("saved to", path+name+".png")

    def plt_single_img(self, img, save=False, path=None, name=None):
        plt.imshow(img)
        if save:
            plt.savefig( path+name)
            #print("saved to", path+name+".png")

    def save_plot_img(self, img):
        if self.plt_fig is None:
            self.plt_obj = []
            self.plt_fig = plt.figure()
            self.plt_obj.append(plt.imshow(img))
            plt.show(block=False)
            #plt.savefig("~/test_saving/save_" + str(current_iteration_step) + ".png")
        else:
            self.plt_obj[0].set_data(img)
            self.plt_fig.canvas.draw()
            #plt.savefig("~/test_saving/save_" + str(current_iteration_step) + ".png")
        return

    def plotImage(self, imgs):
        plt.ion()
        plt.show()
        if self.first_time:
            for idx, im in enumerate(imgs):
                #if idx == 0:
                #    im = cv2.putText(np.array(im), "Entropy: " + str(self.sim.entropy_history[0]), (50, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 1, 2, cv2.LINE_AA)
                plt.subplot(1, len(imgs), idx+1)
                self.plt_obj.append(plt.imshow(im))
                plt.pause(0.001)
            self.first_time = False

        else:
            for idx, im in enumerate(imgs):
                #if idx == 0:
                #    im = cv2.putText(np.array(im), "Entropy: " + str(self.sim.entropy_history[0]), (50, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 1, 3, cv2.LINE_AA)
                self.plt_obj[idx].set_data(im)
            plt.pause(0.001)

        return


    def plotImageold(self, imgs):
        if self.plt_fig is None:
            self.plt_obj = []
            self.plt_fig = plt.figure()
            # Plot truth
            if imgs is not None:
                plt.subplot(1, 1, 1)
                self.plt_obj.append(plt.imshow(imgs))
                #plt.axis("off")
                plt.show(block=False)

        else:
            mng = plt.get_current_fig_manager()
            mng.resize(800,800)
            if imgs is not None:
                self.plt_obj[0].set_data(imgs)
            self.plt_fig.canvas.draw()
        return
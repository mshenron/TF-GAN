import tensorflow as tf
from nets import discriminator,genrator
import numpy as np
from ops import loss
from random import shuffle
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging


logger=logging.getLogger('Dcgan')
logger.setLevel(logging.INFO)

handler=logging.FileHandler('__model__.log')
handler.setLevel(logging.INFO)

formatter=logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)

class dcgan():
    def __init__(self,img_shape,model_path,logdir,sampledir,epochs=200,gen_lr=0.001,dis_lr=0.001,z_shape=100,batch_size=64,
                    beta1=0.5,SampleAfter=100,SaveAfter=1000):
        ## Loading parameters so other methods can access them easily
        self.height,self.width,self.channels=img_shape
        self.epochs=epochs
        self.gen_lr=gen_lr
        self.dis_lr=dis_lr
        self.z_shape=z_shape #single integer value
        self.batch_size=batch_size
        self.beta1=beta1
        self.SampleAfter=SampleAfter
        self.SaveAfter=SaveAfter
        self.model_path=model_path
        self.logdir=logdir
        self.sampledir=sampledir

        # Initiating genrator and discriminator object
        self.genrator=genrator(img_shape,z_shape)
        self.discriminator=discriminator(img_shape)

        # Loading Dataset
        mnist=tf.keras.datasets.mnist
        (x_train,_),(x_test,_)=mnist.load_data()
        x_train=np.concatenate([x_train,x_test])
        self.x_train=x_train/127.5-1

        ## Input placeholders
        self.in_x=tf.placeholder(tf.float32,[None,self.height,self.width])
        self.in_z=tf.placeholder(tf.float32,[None,z_shape])

        ## genrate images 
        self.genrated=self.genrator.feed(self.in_z)
        ## Feeding both fake and real images into discriminator
        DisFake=self.discriminator.feed(self.genrated)
        DiscReal=self.discriminator.feed(self.in_x)

        ## Calculating loss , trying to predict genrated images as fake and real images as real
        FakeLoss=loss(tf.zeros_like(DisFake),DisFake)
        RealLoss=loss(tf.ones_like(DiscReal),DiscReal)
        
        #Defining genrator and discriminator loss
        self.DisLoss=tf.add(FakeLoss,RealLoss)
        self.GenLoss=loss(tf.ones_like(DisFake),DisFake)
        ## Adding summary for tensorboard visualization
        tf.summary.scalar("DisLos",self.DisLoss)
        #tf.summary.scalar("GenLoss",self.GenLoss)

        ## Seprating descriminator and genrator trainable variables
        TrainVar=tf.trainable_variables()
        DisVar=[var for var in TrainVar if 'DIS' in var.name]
        GenVar=[var for var in TrainVar if 'GEN' in var.name]

        self.DisOpt=tf.train.AdamOptimizer(self.dis_lr,self.beta1).minimize(self.DisLoss,var_list=DisVar)
        self.GenOpt=tf.train.AdamOptimizer(self.gen_lr,self.beta1).minimize(self.GenLoss,var_list=GenVar)

        self.SummaryOp=tf.summary.merge_all()
        self.saver=tf.train.Saver()

    def train(self):
        init=tf.global_variables_initializer()
        self.sess=tf.Session()
        ## Initialize graph variables 
        self.sess.run(init)
        summary_writer=tf.summary.FileWriter(self.logdir,graph=tf.get_default_graph())
        ## Restore existing checkpoint
        if os.path.exists("//".join(model_path.split('/')[:-1])+'//checkpoint'):
            self.saver.restore(self.sess,self.model_path)
            logger.info(f'Restored Model from {self.model_path}')
        ## Creating Progress bar object
        pbar=tqdm(self.epochs*(len(self.x_train)//self.batch_size))

        ## started training
        steps=0
        for e in range(self.epochs): 
            ## create a random list of indices
            RandomIDX=list(range(len(self.x_train)))
            shuffle(RandomIDX)
            pointer=0
            for _ in range(int(len(self.x_train)/self.batch_size)):
                # random sample of images using shuffled indices 
                x=self.x_train[RandomIDX[pointer:pointer+self.batch_size]]
                pointer+=self.batch_size
                # random input for genrator
                z=np.random.uniform(-1, 1, (self.batch_size, self.z_shape))
                _,dis_loss,summary=self.sess.run([self.DisOpt,self.DisLoss,self.SummaryOp],feed_dict={self.in_x:x,self.in_z:z})
                summary_writer.add_summary(summary, steps)
                # random input for genrator
                z=np.random.uniform(-1, 1, (self.batch_size, self.z_shape))
                _,gen_loss=self.sess.run([self.GenOpt,self.GenLoss],feed_dict={self.in_z:z})
                if steps%self.SampleAfter==0:
                    self.create_samples(steps)
                    logger.info(f"Loss after {e} epochs & {steps} steps : \t Genrator Loss : {gen_loss} \t Discriminator Loss : {dis_loss}")
                if steps%self.SaveAfter==0:
                    save_path=self.saver.save(self.sess,self.model_path)
                    logger.info(f'Model saved at path {save_path}')
                pbar.update(1)
                steps+=1
        pbar.close()

    def create_samples(self,steps):
        col,row = (7,7)
        z = np.random.uniform(-1, 1, (self.batch_size, self.z_shape))
        imgs = self.sess.run(self.genrated, feed_dict={self.in_z:z})
        imgs = imgs*0.5 + 0.5
        # scale between 0, 1
        fig, axs = plt.subplots(col, row)
        cnt = 0
        for i in range(col):
            for j in range(row):
                axs[i, j].imshow(imgs[cnt, :, :, 0], cmap="gray")
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig( f"{self.sampledir}//GenAfter_{steps}_steps.png")
        plt.close()


if __name__=="__main__":
    img_shape=(28,28,1)
    epochs=200
    logdir=r".//tmp//tf_log//"
    sampledir=r'.//tmp//sample//'
    model_path=r'./tmp/checkpoint/model.ckpt'
    model_dir="//".join(model_path.split('/')[:-1])
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    if not os.path.exists(sampledir):
        os.makedirs(sampledir)

    model=dcgan(img_shape=img_shape,model_path=model_path,sampledir=sampledir,logdir=logdir,epochs=epochs)
    model.train()

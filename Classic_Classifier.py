#database/csv libraries
import cv2
import os
import shutil
import seaborn as sns
import pandas as pd
import numpy as np

#imagery libraries
from scipy import ndimage
from torchvision import transforms
import matplotlib.pyplot as plt

#classic CNN libraries
import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping 
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D 
from tensorflow.python.keras.layers import Flatten, Dense 
from tensorflow.python.keras.models import Sequential
from keras import layers 
from keras.regularizers import l2

#Quantum qpeg library
import torch





#initialising path to images
path = 'images/'

#check if GPU is available for use in classification
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#ask if images are moved to respective classes
done = input("are images moved yet?")


#enable options to overwrite dataframe
pd.options.mode.copy_on_write = True




#isolating the useful information from the csv files and shuffling the output.
def Dataset():

        
        #read csv file containing image asset_id and galaxy object id, isolating these columns using pandas
        dftemp1 = pd.read_csv("gz2_filename_mapping.csv")    
        df1 = pd.DataFrame({'asset_id':dftemp1.asset_id,'objid':dftemp1.objid})
        
        #read csv file containing galaxy object id and galaxy zoo 2 classification, isolating these columns using pandas
        dftemp2 = pd.read_csv("gz2_hart16.csv")
        df2 = pd.DataFrame({'objid':dftemp2.dr7objid,'gz2_class':dftemp2.gz2_class,
                            'Fraction_Vote_DiskFeature':dftemp2.t01_smooth_or_features_a02_features_or_disk_fraction,
                            'Fraction_Vote_No':dftemp2.t01_smooth_or_features_a01_smooth_fraction,
                            'Debiased_Fraction_Vote_DiskFeature':dftemp2.t01_smooth_or_features_a02_features_or_disk_debiased,
                            'Debiased_Fraction_Vote_No':dftemp2.t01_smooth_or_features_a01_smooth_debiased})
        
        #merging these two dataframes in to 1 containing Image asset id, galaxy object id and galaxy zoo 2 classification
        #and outputting it in a seperate csv file for later use.
        df3 = pd.merge(df1, df2)
        #df3.set_index('asset_id', inplace=True)
        
        

        #remove entries for which no image exists in database
        #CAN ONLY BE DONE ONCE, after images are redirected towards their
        #directory this function will not work anymore
        

        i=0
        while i < len(df3):
            #loop over all asset id's and check if in directory
            #get corresponding asset id
            check = df3.iloc[i]['asset_id']
            #checks if path file exists
            exist = os.path.isfile(f'{path}{check}.jpg')
            if exist == False:
                #if there is no file path it is dropped from the dataframe
                df3 = df3.drop(axis = 0, index=i).reset_index(drop=True)
                #check immediate following in case it is missed due to dataframe being compressed
                i = i - 1
            i=i+1#increment counter

            
        #export dataframe to csv file for easier inspection
        df3['main_type'] = df3['gz2_class'].astype(str).str[0]
        df4 = df3.loc[df3['main_type'] != 'A']
        df4.to_csv('Database_Galaxies.csv')#contains dataframe of imagery, binary classification and vote fractions for baseline 
            
        return 0

    
#full dataset not used right now + data seperation also not used rn
#Dataset()

#df4 = pd.read_csv("database.csv") #changed to smallbase for now
#df4 = pd.read_csv("smallbase.csv")
#df5 = pd.read_csv("Database_Galaxies.csv")
#dfsmall = df5.drop_duplicates(keep='first')



def Baseline_accuracy():
    df5 = pd.read_csv("Database_Galaxies.csv")
    dfnew = df5.drop_duplicates(keep='first')

    #dfsmall['main_type'] = dfsmall['gz2_class'].astype(str).str[0]
    #remove all A classed objects
    #dfnew = dfsmall.loc[dfsmall['main_type'] != 'A']

    Galaxy_classes = dfnew['main_type'].unique()
    print(Galaxy_classes)
    #for every category, 
    for cat in Galaxy_classes:
        x = dfnew.loc[dfnew['main_type'] == cat]
        i=0
        total_vote = 0
        total_vote_deb = 0
        #check fraction of no Disked Galaxies for E
        if cat == 'E':
            while i < len(x):
                #count biased fraction of votes
                Fraction_Vote = x.iloc[i]['Fraction_Vote_No']
                total_vote += Fraction_Vote
                #count debiased fraction of votes
                Fraction_Vote_deb = x.iloc[i]['Debiased_Fraction_Vote_No']
                total_vote_deb += Fraction_Vote_deb


                i=i+1
            Average_Fraction_E = total_vote / len(x)
            Average_Fraction_E_deb = total_vote_deb / len(x)
            
            

        #and check fraction of Disked Galaxies for S
        if cat == 'S':
            while i < len(x):
                #count biased fraction of votes
                Fraction_Vote = x.iloc[i]['Fraction_Vote_DiskFeature']
                total_vote += Fraction_Vote
                #count debiased fraction of votes
                Fraction_Vote_deb = x.iloc[i]['Debiased_Fraction_Vote_DiskFeature']
                total_vote_deb += Fraction_Vote_deb

                i=i+1
            Average_Fraction_S = total_vote / len(x)
            Average_Fraction_S_deb = total_vote_deb / len(x)
            
    print("Average fraction voted on smooth galaxy for elliptical classified galaxies =", Average_Fraction_E)
    print("Average fraction voted on spiraled galaxy for spiraled classified galaxies =", Average_Fraction_S)
    print("Average debiased fraction voted on smooth galaxy for elliptical classified galaxies =", Average_Fraction_E_deb)
    print("Average debiased fraction voted on spiraled galaxy for spiraled classified galaxies =", Average_Fraction_S_deb)

    return 0

#Baseline_accuracy()

def binary_directory(): #CHANGE SMALLIMAGES WITH TEMP WHEN DONE
    df5 = pd.read_csv("Database_Galaxies.csv")
    dfsmall = df5.drop_duplicates(keep='first')

    dfsmall['main_type'] = dfsmall['gz2_class'].astype(str).str[0]

    Galaxy_classes = dfsmall['main_type'].unique()
    if not done == 'ja':
    #moves all imagery into their classes directory wise
    #this is easier for image classification.
        for cat in Galaxy_classes:
            x = dfsmall.loc[dfsmall['main_type'] == cat]
            pathcreatdir = (f'truncimages/{cat}')

            #check if this has already been done before to reduce time complexity
            doesexist = os.path.exists(pathcreatdir)
            if not doesexist:
                os.makedirs(pathcreatdir)

            #for every image per class move it to their respective directory 
            #also after checking if this was already done before
            i=0
            while i < len(x):
                image_number = x.iloc[i]['asset_id']
                image_exist = os.path.exists(f'truncimages/{cat}/{image_number}.jpg')
                if not image_exist:
                    shutil.copy(f'{path}{image_number}.jpg',f'truncimages/{cat}')
                    #Convert L for grayscale mode, .resize((4, 4)) for resize to 4x4
                    #img_rgb = Image.open(f'smallimages/{cat}/{image_number}.jpg').convert('L').resize((4, 4))
                    #img_rgb.save(f'smallimages/{cat}/{image_number}.jpg')
                i=i+1

    return 0


#binary_directory()

def calc_mean():
    df5 = pd.read_csv("Database_Galaxies.csv")
    dfsmall = df5.drop_duplicates(keep='first')

    dfnew = dfsmall.loc[dfsmall['main_type'] != 'A']

    Galaxy_classes = dfnew['main_type'].unique()
    print(Galaxy_classes)
    dfnew["mean_intensity"] = np.nan

    i = 0
    while i < len(dfnew):
        image_number = dfnew.iloc[i]['asset_id']
        EORS = dfnew.iloc[i]['main_type']
        vol = cv2.imread(f'tempimages/{EORS}/{image_number}.jpg')
        mean_all = ndimage.mean(vol, labels=None, index=None)
        dfnew.iloc[i, dfnew.columns.get_loc('mean_intensity')] = mean_all

        i=i+1



    for cat in Galaxy_classes:
        x = dfnew.loc[dfnew['main_type'] == cat]
        sns.kdeplot(x['mean_intensity'],label=cat)
    
    plt.legend()
    #plt.legend(prop={'size': 16}, title = 'galaxy class')
    plt.xlabel('mean intensity image')
    plt.ylabel('Density')
    plt.show()


def calc_variance():
    df5 = pd.read_csv("Database_Galaxies.csv")
    dfsmall = df5.drop_duplicates(keep='first')

    dfnew = dfsmall.loc[dfsmall['main_type'] != 'A']

    Galaxy_classes = dfnew['main_type'].unique()
    print(Galaxy_classes)
    dfnew["mean_variance"] = np.nan

    i = 0
    while i < len(dfnew):
        image_number = dfnew.iloc[i]['asset_id']
        EORS = dfnew.iloc[i]['main_type']
        vol = cv2.imread(f'tempimages/{EORS}/{image_number}.jpg')
        variance_all = ndimage.variance(vol, labels=None, index=None)
        dfnew.iloc[i, dfnew.columns.get_loc('mean_variance')] = variance_all

        i=i+1



    for cat in Galaxy_classes:
        x = dfnew.loc[dfnew['main_type'] == cat]
        sns.kdeplot(x['mean_variance'],label=cat)
    
    plt.legend()
    #plt.legend(prop={'size': 16}, title = 'galaxy class')
    plt.xlabel('mean variance image')
    plt.ylabel('Density')
    plt.show()



#calc_mean()

#put images back into original order for other classification method that needs other classes
def undo_directory_images():
    #WINDOWS does not allow for ? in file folders, fixing this by replacing all question marks with exclamation marks!
    #for undoing images ? also needs to be replaced with !
    df5 = pd.read_csv("Database_Galaxies.csv")
    dfsmall = df5.drop_duplicates(keep='first')

    dfsmall['gz2_class'] = dfsmall['gz2_class'].str.replace('?','!')
     #gz2_class for undoing 818 classes, main type for 3 classes
    Galaxy_classes = dfsmall['main type'].unique()


    for cat in Galaxy_classes:
        #gz2_class for undoing 818 classes, main type for 3 classes
        x = dfsmall.loc[dfsmall['main type'] == cat]
        

        i=0
        while i < len(x):
                image_number = x.iloc[i]['asset_id']

                image_exist = os.path.exists(f'tempimages/{cat}/{image_number}.jpg')
                if image_exist:
                    shutil.move(f'tempimages/{cat}/{image_number}.jpg',f'{path}')
                i=i+1
        
        os.rmdir(f'tempimages/{cat}')
    return 0


def Classic_CNN_model(correct_dir):
    df5 = pd.read_csv("Database_Galaxies.csv")
    dfsmall = df5.drop_duplicates(keep='first')

    dfsmall['main_type'] = dfsmall['gz2_class'].astype(str).str[0]
    dfnew = dfsmall.loc[dfsmall['main_type'] != 'A']

    Galaxy_classes = dfnew['main_type'].unique()
    print(Galaxy_classes)
    
    
    directory_exist = os.path.exists(f'tempimages/A')
    if directory_exist:
        shutil.move(f'tempimages/A', os.getcwd())

    #DATA = 'tempimages'
    #DATA = f'smalltruncimages{correct_dir}'
    DATA = f'{correct_dir}'

    print(DATA)

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(DATA, validation_split=0.2, 
                                                                subset='training', 
                                                                image_size=( 
                                                                    424, 424), 
                                                                seed=123, 
                                                                batch_size=64) 
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(DATA, validation_split=0.2, 
                                                                subset='validation', 
                                                                image_size=( 
                                                                    424, 424), 
                                                                seed=123, 
                                                                batch_size=64) 


    #DATA2 = f'smalltruncimagesgray'
    
    #test_ds = tf.keras.preprocessing.image_dataset_from_directory(DATA2,
    #                                                              image_size=( 
    #                                                                424, 424))

    class_names = train_ds.class_names 
    print(class_names)



    #data_augmentation = tf.keras.Sequential( 
    #    [ 
    #        tf.keras.layers.experimental.preprocessing.RandomFlip( 
    #            "horizontal", input_shape=(424, 424, 3)), 
    #        tf.keras.layers.experimental.preprocessing.RandomRotation(0.1), 
    #        tf.keras.layers.experimental.preprocessing.RandomZoom(0.2), 
    #        tf.keras.layers.experimental.preprocessing.RandomFlip( 
    #            mode="horizontal_and_vertical") 
    #    ] 
    #) 

    #model for full database
    model = Sequential() 
    model.add(layers.Rescaling(1./255)) 
    model.add(Conv2D(32, (10, 10), activation='relu')) 
    model.add(MaxPooling2D((2, 2))) 
    model.add(Conv2D(16, (20, 20), activation='relu')) 
    model.add(MaxPooling2D((20, 20))) 
    model.add(Flatten()) 
    model.add(Dense(len(class_names), activation='softmax', activity_regularizer=l2(0.01))) 

    #model for 15 singular value database
    #this is to check if this model could learn with the first 15  eigenvalues, but not with the full image:
    #model = Sequential() 
    #model.add(Conv2D(32, (10, 10), activation='relu')) 
    #model.add(MaxPooling2D((20, 20))) 
    #model.add(Flatten()) 
    #model.add(Dense(len(class_names), activation='softmax', activity_regularizer=l2(0.01))) 


    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
    mycallbacks = [EarlyStopping(monitor='val_loss', patience=50)] 
    history = model.fit(train_ds,
                    #steps_per_epoch =50,
                    validation_data=val_ds, 
                    epochs=8,
                    callbacks=mycallbacks) 
    print(model.summary())
    #y_pred = model.predict(val_ds)
    #print(y_pred)
    #model.evaluate(test_ds)

    # Loss 
    plt.subplot(121)
    plt.plot(history.history['loss']) 
    plt.plot(history.history['val_loss']) 
    plt.legend(['loss', 'val_loss'], loc='upper right') 

    # Accuracy 
    plt.subplot(122)
    plt.plot(history.history['accuracy']) 
    plt.plot(history.history['val_accuracy']) 
    plt.legend(['accuracy', 'val_accuracy'], loc='upper right') 
    plt.show()

    #Y_pred = model.predict(val_ds)
    #Y_pred = np.argmax(Y_pred, axis=1)

    #metrics.confusion_matrix(val_ds, Y_pred)

    #print(metrics.classification_report(val_ds, Y_pred,
     #                                   target_names=class_names))
    return 0


#Classic_CNN_model()

def SVD_Truncation():
    df5 = pd.read_csv("Database_Galaxies.csv")
    dfsmall = df5.drop_duplicates(keep='first')

    dfsmall['main_type'] = dfsmall['gz2_class'].astype(str).str[0]
    Galaxy_classes = dfsmall['main_type'].unique()
    print(Galaxy_classes)

    var_total = 0
    comps = [5, 10, 15, 20]

    if not done == 'ja':
    
        for cat in Galaxy_classes:
            
            if cat == 'E':
                i=0
                x = dfsmall.loc[dfsmall['main_type'] == cat]
                while i < len(x):
                    
                    image_number = x.iloc[i]['asset_id']
                    
                    image = cv2.imread((f'images/{image_number}.jpg'))
                    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
                    u, s, v = np.linalg.svd(gray_image, full_matrices=False)

                    #save normal gray image
                    pathcreategray = (f'Full_Image_Gray/{cat}')
                    #check if this has already been done before to reduce time complexity
                    doesexist = os.path.exists(pathcreategray)
                    if not doesexist:
                        os.makedirs(pathcreategray)
                    cv2.imwrite(f'Full_Image_Gray/{cat}/{image_number}.jpg', gray_image)

                    #convert images to certain ranks.
                    #5 10 15 20

                    for j in range(len(comps)): 
                        low_rank = u[:, :comps[j]] @ np.diag(s[:comps[j]]) @ v[:comps[j], :] 

                        pathcreatdir = (f'SVD{comps[j]}/{cat}')

                        #check if this has already been done before to reduce time complexity
                        doesexist = os.path.exists(pathcreatdir)
                        if not doesexist:
                            os.makedirs(pathcreatdir)

                        cv2.imwrite(f'SVD{comps[j]}/{cat}/{image_number}.jpg', low_rank)
                    

                    
                    print(i/len(x))
                    i=i+1

            
                    
            #BELOW IS VARIANCE EXPLAINED, IMAGES ARE FOUND IN FOLDER AND THUS FOR NOW EVERYTHING
            #IS COMMENTED TO REDUCE CALC TIME
                    

            #var_explained = np.round(s**2/np.sum(s**2), decimals=6)                

            #var_total += var_explained                                             
            #print(i/len(x))
            

    #var_average = var_total / len(dfsmall)                                         
  
    # Variance explained top Singular vectors 
    #print(f'variance Explained by Top 20 singular values:\n{var_average[0:20]}')   
  
    #sns.barplot(x=list(range(1, 21)), 
    #   y=var_average[0:20], color="dodgerblue")                                    
  
    #plt.title('Variance Explained Graph') 
    #plt.xlabel('Singular Vector', fontsize=16)
    #plt.yscale('log')                                     
    #plt.ylabel('Variance Explained', fontsize=16) 
    #plt.tight_layout() 
    #plt.show() 



    #old pipelined way of faSTER   checking model
    #gray = 'gray'
    #call neural network to classify
    #Classic_CNN_model(gray)
    #for k in range(len(comps)):
    #    Classic_CNN_model(comps[k])


    return 0

def QPEGCOMPRESSION():
    df5 = pd.read_csv("Database_Galaxies.csv")
    dfsmall = df5.drop_duplicates(keep='first')

    #functions for converting images to tensors and back
    convert_tensor = transforms.ToTensor()
    transform = transforms.ToPILImage()

    dfsmall['main_type'] = dfsmall['gz2_class'].astype(str).str[0]
    dfnew = dfsmall.loc[dfsmall['main_type'] != 'A']
    Galaxy_classes = dfnew['main_type'].unique()
    print(Galaxy_classes)
    comps = [5, 10, 15, 20] 

    if not done == 'ja':
    
        for cat in Galaxy_classes:
            
            if cat == 'S':
                i=0
                x = dfnew.loc[dfnew['main_type'] == cat]
                while i < len(x):
                    
                    image_number = x.iloc[i]['asset_id']
                    image = cv2.imread((f'images/{image_number}.jpg'))

                    #turn image gray
                    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

                    #store image in a tensor node
                    node = convert_tensor(gray_image)

                    #remove unwanted extra dimensionality that occurs when transforming the image to a tensor.
                    Tnode = node.reshape(-1, node.shape[-1])

                    #decompose tensor with pytorch SVD function
                    U, S, VT = torch.linalg.svd(Tnode)

                    #convert images to certain ranks.
                    #5 10 15 20

                    for j in range(len(comps)): 
                        low_rank = U[:, :comps[j]] @ np.diag(S[:comps[j]]) @ VT[:comps[j], :] 
                        TensorRank = transform(low_rank)
                        TensorImage = np.array(TensorRank)
                        pathcreatdir = (f'Tensor{comps[j]}/{cat}')
                        
                        #check if this has already been done before to reduce time complexity
                        doesexist = os.path.exists(pathcreatdir)
                        if not doesexist:
                            os.makedirs(pathcreatdir)
                        
                        cv2.imwrite(f'Tensor{comps[j]}/{cat}/{image_number}.jpg', TensorImage)
                    

                    
                    print(i/len(x))
                    i=i+1

    return 0


#SVD_Truncation()

#perform model on origninal grayscaled database
#Classic_CNN_model('Full_Image_Gray')


#call normal SVD compressed images
#Classic_CNN_model('SVD5')
#Classic_CNN_model('SVD10')
#Classic_CNN_model('SVD15')
#Classic_CNN_model('SVD20')

#call normal Tensor SVD compressed images
#Classic_CNN_model('Tensor5')
#Classic_CNN_model('Tensor10')
#Classic_CNN_model('Tensor15')
#Classic_CNN_model('Tensor20')


#QPEGCOMPRESSION()

def main():

    choice ='0'
    while choice =='0':
        print("Main Choice: Choose 1 of 5 choices")
        print("Choose 1 for creating the dataframe used for classification")
        print("Choose 2 for seperating images into respective class from dataframe")
        print("Choose 3 for calculating classification accuracy baseline for comparison")
        print("Choose 4 for calculating mean intensity of the image dataset")
        print("Choose 5 for calculating mean variance of the image dataset")
        print("Choose 6 to choose compression method")
        print("Choose 7 to start classification algorithm")

        choice = input ("Please make a choice: ")

        if choice == "1":
            print("dataframe is being created and saved to a seperate csv file...")
            Dataset()

        elif choice == "2":
            print("Do Something 4")

        elif choice == "3":
            print("Do Something 3")

        elif choice == "4":
            print("Do Something 2")

        elif choice == "5":
            print("Do Something 1")

        elif choice == "6":
            print("Do Something 1")

        elif choice == "7":
            print("Do Something 1")
            
        else:
            os.system( 'cls' )
            print("I don't understand your choice.")
            main()

def second_menu():
     print("This is the second menu")


main()
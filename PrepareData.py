import os.path

import pandas
import Config
import Utilty

Utilty.IsNormalised=Config.NormalizedFlag

# This file is genrating all features for user in both format and also generating user key

person=Config.users
Genuine=Config.Genuine
Fake=Config.Fake

def GenerateRandomKeyForUsers():
    users=Config.users
    taps=[15, 14, 5, 4]
    col=['User','Key']
    val=[]
    for user in users:
        temp_k=Utilty.generate_random_number()
        key=Utilty.lfsr(temp_k,taps,Config.KeySize)
        val.append([user,key])
    df=pandas.DataFrame(val,columns=col)
    df.to_csv(Config.key_file_name,index=False)


def get_users_file_path(p):
    try:
        #for p in person:
        user_file_path = []
        for sign in Genuine:
            path=Config.db_root+'USER'+str(p)+'_'+str(sign)+Config.data_file_type
            user_file_path.append(path)
        for sign in Fake:
            path = Config.db_root + 'USER' + str(p) + '_' + str(sign) +Config.data_file_type
            user_file_path.append(path)
    except:
        print('Error in PrepareData.py in get_users_file_path() method')
    return user_file_path

def get_features():
    features={}
    for p in person:
        user_file_path = get_users_file_path(p)
        feature=[]
        for path in user_file_path:
            if os.path.exists(path):
                feature.append(Utilty.get_all_features(path, True))
        features[p]=feature
    return features

def get_all_user_data():
    features=get_features()
    formated_data={}
    temp_data={}
    for p in person:
        all_data=features[p]
        X=[];Y=[];T=[];B=[];Az=[];Al=[];P=[];

        path_tangent_angle=[];path_velocity=[];log_curvature_radius=[];total_acceleration=[];

        for file in all_data:
            X.append(file[0])
            Y.append(file[1])
            T.append(file[2])
            B.append(file[3])
            Az.append(file[4])
            Al.append(file[5])
            P.append(file[6])
            path_tangent_angle.append(file[7])
            path_velocity.append(file[8])
            log_curvature_radius.append(file[9])
            total_acceleration.append(file[10])

        temp_data['X']=X
        temp_data['Y'] = Y
        temp_data['T'] = T

        temp_data['B'] = B
        temp_data['Az'] = Az
        temp_data['Al'] = Al

        temp_data['P'] = P
        temp_data['path_tangent_angle'] = path_tangent_angle
        temp_data['path_velocity'] = path_velocity

        temp_data['log_curvature_radius'] = log_curvature_radius
        temp_data['total_acceleration'] = total_acceleration

        formated_data[p]=temp_data

    return formated_data
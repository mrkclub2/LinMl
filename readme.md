LinMl
=====   
Machine Learning Api based Library 

Features
=========
- API auth using tokens
- Monitors
- Configs
- Direct access to ML algorithms
 'sequence': [{
                                # if max_size is specified, not matter what
                                # the resize value in stream_options, it will be rescaled down to this
                                # value if needed
                                'max_size':800,
                                'face_detection_framework': 'dlib',
                                'known_images_path': '/var/lib/zmeventnotification/known_faces',
                                'face_model': 'cnn',
                                'face_train_model': 'cnn',
                                'face_recog_dist_threshold': 0.6,
                                'face_num_jitters': 1,
                                'face_upsample_times':1
                            }]
                        },
## ALPR config
 can be used with mlapi or pyzm 


        'alpr': {
           'general':{
                'same_model_sequence_strategy': 'first',
                'pre_existing_labels':['car', 'bus', 'truck']

          },

        'sequence': [{
            'alpr_api_type': 'cloud',
            'alpr_service': 'plate_recognizer',
            'alpr_url':'http://127.0.0.1:8000/api/v1',
            'alpr_key': utils.get(key='PLATEREC_ALPR_KEY', section='secrets', conf=conf),
            'platrec_stats': 'no',
            'platerec_min_dscore': 0.1,
            'platerec_min_score': 0.2,
         }]
    }



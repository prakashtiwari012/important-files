import mlflow
from urllib.parse import urlparse
from mlflow.tracking import MlflowClient
import os

#%%
path= r"/opt/app/mpidatasci/prakash.tiwari/UB/"
extension = r"/mlruns"

#%%
def check_dir_exist(folder_name):
    total_path = path+folder_name
    try:
        if not os.path.exists(total_path):
            os.makedirs(total_path)
            return True
        else:
            return True
    except:
        return False
#%%
def log_experiment( experiment,
                    hyper_para,
                    eval_metrics,
                    model,
                    tags,
                    experiment_tag,
                    run_tag,
                    folder_name, 
                    artifact_path = 'None',
                    notebook_path = 'None'
                ):
    
    print('Artifact path :' , artifact_path)
    print('notebook_path :' , notebook_path)

    if check_dir_exist(folder_name):
        
        tracking_uri = path+folder_name+extension
        # set experiment
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("/"+str(experiment))

        with mlflow.start_run() as run:
            print("Set up of MLFLOW experimet done and mlflow started")
            run_id = run.info.run_uuid
            experiment_id = run.info.experiment_id

            print(f"Run_id:{run_id} and Experiment_id:{experiment_id}")

            # mlflow client 
            client = MlflowClient()

            client.set_experiment_tag(experiment_id, "mlflow.note.content", experiment_tag)
            client.set_tag(run_id, "mlflow.note.content", run_tag)

            print('Logging started')
            # log hyper-parameters
            mlflow.log_params(hyper_para)

            # log evaluation metrics
            mlflow.log_metrics(eval_metrics)

            print(f"Tracking URI : {mlflow.get_tracking_uri()}")
            # tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            # print('type score',tracking_url_type_store)

            # log tags
            mlflow.set_tags(tags)

            mlflow.sklearn.log_model(model, "model")
            try:
                print('Logging artifacts')
                mlflow.log_artifact(artifact_path,artifact_path="")
            except Exception as e:
                print('Cannot log the artifacts!!', e)
            print('Logging completed')
            mlflow.end_run()
        
    return True
    
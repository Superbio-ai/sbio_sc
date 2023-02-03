
from config_utils import parse_config
import json
from app_runner_utils import AppRunnerUtils
import logging
import sys
import traceback
import time


def train_model(config):
       
    adata = load_data(config)
    logging.info("Data loading completed")
    

    results_for_upload, results_for_payload = generate_response(tsne_list, plot_list, table_list, file_locs,
                                                                test_list, test_locs, [download_list], qcplot, config)
    logging.info("JSON rsponse generated")
    
    return results_for_upload, results_for_payload
    
    
def main():
    JOB_LOG_FILE = 'job.log'
    AppRunnerUtils.set_logging(JOB_LOG_FILE)
    job_id = sys.argv[1]
    logging.info(f'Job id: {job_id}')
    try:
        pre_config = AppRunnerUtils.get_job_config(job_id)
        logging.info(f'Job config: {pre_config}')
        AppRunnerUtils.set_job_running(job_id)
        logging.info(f'Job {job_id} is running')
        config = parse_config(pre_config)
        logging.info("Config parsed and defaults set where necessary.")
        t1_start = time.time() 
        if config['app_name']=='diff_exp':
            results_for_upload, results_for_payload = diff_exp(config)
        t1_stop = time.time() 
        logging.info(f'Elapsed time:{t1_stop-t1_start}') 
        # upload results
        AppRunnerUtils.upload_result_files(job_id, results_for_upload)
        # update job is done
        AppRunnerUtils.set_job_completed(job_id, results_for_payload)
    except Exception as e:
        err = str(e)
        AppRunnerUtils.set_job_failed(job_id, err)
        logging.error(traceback.format_exc())
    finally:
        # upload log files to S3
        AppRunnerUtils.upload_file(job_id, JOB_LOG_FILE)

if __name__ == '__main__':
    main()
    '''
    JOB_LOG_FILE = 'job.log'
    AppRunnerUtils.set_logging(JOB_LOG_FILE)
    with open('/app/tests/diff_exp_test_config.txt') as f:
    #with open('/app/tests/autozi_test_config.txt') as f:
        json_data = json.load(f)
    config = parse_config(json_data)
    results_for_upload, results_for_payload = diff_exp(config)
    print(results_for_upload)
    print(results_for_payload)
    '''
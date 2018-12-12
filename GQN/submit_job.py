training_inputs = {'scaleTier': 'CUSTOM',
    'masterType': 'complex_model_m',
    'workerType': 'complex_model_m',
    'parameterServerType': 'large_model',
    'workerCount': 9,
    'parameterServerCount': 3,
    'packageUris': ['gs://my/trainer/path/package-0.0.0.tar.gz'],
    'pythonModule': 'trainer.task'
    'args': ['--arg1', 'value1', '--arg2', 'value2'],
    'region': 'us-central1',
    'jobDir': 'gs://my/training/job/directory',
    'runtimeVersion': '1.10',
    'pythonVersion': '3.5'}
job_spec = {'jobId': "run55", 'trainingInput': training_inputs}


project_name = 'gqnmodel'
project_id = 'projects/{}'.format(project_name)

cloudml = discovery.build('ml', 'v1')

request = cloudml.projects().jobs().create(body=job_spec,
              parent=project_id)
response = request.execute()

try:
    response = request.execute()
    # You can put your code for handling success (if any) here.

except errors.HttpError, err:
    # Do whatever error response is appropriate for your application.
    # For this example, just send some text to the logs.
    # You need to import logging for this to work.
    logging.error('There was an error creating the training job.'
                  ' Check the details:')
    logging.error(err._get_reason())

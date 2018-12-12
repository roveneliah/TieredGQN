from googleapiclient import discovery

projectName = 'pure-gqn'
projectId = 'projects/{}'.format(projectName)
jobName = 'run54'
jobId = '{}/jobs/{}'.format(projectId, jobName)

ml = discovery.build('ml','v1')
request = ml.projects().jobs().get(name=jobId)

response = None

response = request.execute()

# try:
#     response = request.execute()
# except errors.HttpError, err:
#     # Something went wrong. Handle the exception in an appropriate
#     #  way for your application.
#     print("some error")

if response == None:
    # Treat this condition as an error as best suits your
    # application.)
    print("no response")


print('Job status for {}.{}:'.format(projectName, jobName))
print('    state : {}'.format(response['state']))
print('    consumedMLUnits : {}'.format(
    response['trainingOutput']['consumedMLUnits']))
print(response)

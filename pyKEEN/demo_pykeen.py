from pykeen.pipeline import pipeline

pipeline_result = pipeline(
    dataset='Nations',
    model='TransE',
)
pipeline_result.save_to_directory('nations_transe')

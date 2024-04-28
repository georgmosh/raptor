import os
from raptor import RetrievalAugmentation, RetrievalAugmentationConfig
from raptor.cluster_tree_builder import ClusterSpatialConfig

os.environ["OPENAI_API_KEY"] = "I_LOVE_LASAGNA"

# Cinderella story defined in sample.txt
with open('demo/sample.txt', 'r') as file:
    text = file.read()

tree_builder_config = ClusterSpatialConfig(
    tokenizer=None,
    max_tokens=100,
    num_layers=5,
    threshold=0.5,
    top_k=5,
    selection_mode="top_k",
    summarization_length=100,
    summarization_model=None,
    embedding_models=None,
    cluster_embedding_model="OpenAI",
)

retrieval_augmentation_config = RetrievalAugmentationConfig(tree_builder_config=tree_builder_config)
RA = RetrievalAugmentation(config=retrieval_augmentation_config)

# construct the tree
RA.add_documents(text)

question = "How did Cinderella reach her happy ending ?"
answer = RA.answer_question(question=question)
print("Answer: ", answer)

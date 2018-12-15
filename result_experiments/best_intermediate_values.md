#OLD
1st sector best: 0.0275 {'weights1': 0.45590938562950867, 'weights2': 0.017972928905949592, 'weights3': 0.23905548168035573, 'weights4': 0.017005850670624212, 'weights5': 0.9443556793576228, 'weights6': 0.19081956929601618, 'weights7': 0.11601757370322985, 'weights8': 0.11267140391070507, 'normalize': False}
2nd sector best: 0.0557 'weights1': 0.973259052781316, 'weights2': 0.037386979507335605, 'weights3': 0.8477517414017691, 'weights4': 0.33288193455193427, 'weights5': 0.9696801027638645, 'weights6': 0.4723616073494711, 'weights7': 0.5939341460905799, 'weights8': 0.4188403112229081, 'normalize': False
3rd sector best: 0.0275 {'weights1': 0.6666666666666666, 'weights2': 0.8888888888888888, 'weights3': 0.5555555555555556, 'weights4': 1.0, 'weights5': 0.8888888888888888, 'weights6': 1.0, 'weights7': 1.0, 'weights8': 0.0

Strange recommender parameters:

    lambda_i = 0.1
    lambda_j = 0.05
    old_similrity_matrix = None
    num_factors = 165
    l1_ratio = 1e-06

    'alphaP3': 1.160296393373262,
    'alphaRP3': 0.4156476217553893,
    'betaRP': 0.20430089442930188,

topk = [60, 100, 150, 56, 146, 50, -1, -1]
shrinks = [5, 10, 50, 10, -1, -1, -1, -1]

Recommenders used:
 ItemKNNCBFRecommender,
 UserKNNCBRecommender,
 ItemKNNCFRecommender,
 UserKNNCFRecommender,
 P3alphaRecommender,
 RP3betaRecommender,
 SLIM_BPR_Cython,
 PureSVDRecommender

# NEW!!!
##1st secotor: ItemCBF, ItemCF, UserCf, RP3Beta
{'weights1': 0.6936763453666485, 'weights2': 0.8818900949901204, 'weights3': 0.028286087945956884, 'weights4': 0.9108661028648041
MAP: 0.027608903149289973
Item Collaborative: Best config is: Config {'top1': 595, 'shrink1': 1, 'normalize': False}, MAP value is 0.0237
User Collaborative:  Best config is: Config {'top1': 105, 'shrink1': 30, 'normalize': True}, MAP value is 0.0225
RP3Beta: Config: {'top1': 20, 'alphaRP3': 0.457685370741483, 'betaRP': 0.289432865731463} - MAP results: 0.0242190133616356
Item Content: Best config is: Config {'top1': 15, 'shrink1': 210, 'feature_weighting_index': 0}, MAP value is 0.0189


##3rd sector: ItemCF, UserCF, RP3b
{'weights1': 0.023574800557095155, 'weights2': 0.3954614055660033, 'weights3': 0.4605427159971659}

'alphaRP3': 0.49774549098196397,
'betaRP': 0.2333486973947896,
"topK": [130, 240, 91],
"shrink": [2, 19, -1],
MAP results: 0.028627281415957516

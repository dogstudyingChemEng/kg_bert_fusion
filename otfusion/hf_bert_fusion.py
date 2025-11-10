"""
OTFusion for HuggingFace BERT-style models (KG-BERT)

This file adapts the existing hf_vit_fusion approach to the naming
and nesting conventions used by HF BERT models (e.g. BertForSequenceClassification).
It mostly defines the mapping of keys so the generic otfusion_lib functions
can operate on the nested dicts returned by `model_to_dict`.

(CORRECTED VERSION)
"""
from otfusion_lib import ln_fusion, encoder_fusion, fc_fusion, resid_policy
import copy, logging, torch


def hf_bert_fusion(args: dict, weights: dict, acts: dict, alpha, device: torch.device, LOGGING_LEVEL, log_file = None):
    if log_file != None:
        log = logging.getLogger('{0}_otfusion'.format(log_file))
        fileHandler = logging.FileHandler(log_file, mode='a')
        log.addHandler(fileHandler)
    else:
        log = logging.getLogger('otfusion')
    log.setLevel(LOGGING_LEVEL)

    # initialize fused weight dict with expected structure
    number_of_encoders = len(weights['model_0']['bert']['encoder']['layer'])
    w_fused = {'bert': {'embeddings': {}, 'encoder': {'layer': {}}}}

    # key mapping adapted to huggingface bert naming inside each encoder layer
    keys = {}
    # Encoder keys
    # LayerNorm after attention (attention.output.LayerNorm)
    keys['enc_ln0_keys'] = ['attention', 'output', 'LayerNorm']
    # LayerNorm after feed-forward (output.LayerNorm)
    keys['enc_ln1_keys'] = ['output', 'LayerNorm']
    # attention container
    keys['enc_sa_keys']  = ['attention']
    # FFN keys
    keys['enc_ff0_keys'] = ['intermediate', 'dense']
    keys['enc_ff1_keys'] = ['output', 'dense']

    # Attention keys inside the attention dict
    keys['w_q'] = ['self', 'query']
    keys['w_k'] = ['self', 'key']
    keys['w_v'] = ['self', 'value']
    keys['w_o'] = ['output', 'dense']

    # Fully connected weight/bias names
    keys['weights'] = ['weight']
    keys['bias']    = ['bias']

    # Layer norm param names
    keys['a'] = ['weight']
    keys['b'] = ['bias']

    # --- (CORRECTION 1: T-Map propagation) ---
    # This variable will hold the final transport map from the embedding layer
    # to be passed to the first encoder layer.
    t_out_embed = None

    # Embeddings fusion
    if args['fusion']['fuse_src_embed']:
        log.info(' Fusing word embeddings')
        w_we_0 = weights['model_0']['bert']['embeddings']['word_embeddings']
        w_we_1 = weights['model_1']['bert']['embeddings']['word_embeddings']
        w_we_fused, t_out_embed = fc_fusion(args = args, keys = keys, t_in = None, w_0 = w_we_0, w_1 = w_we_1,
                                    act_0 = acts['model_0']['bert']['embeddings']['word_embeddings'],
                                    act_1 = acts['model_1']['bert']['embeddings']['word_embeddings'],
                                    alpha = alpha, device = device, log = log, last_layer = False, is_embed = True)
        w_fused['bert']['embeddings']['word_embeddings'] = {'weight': w_we_fused['weight'].detach()}

        log.info(' Fusing position embeddings')
        w_pe_0 = weights['model_0']['bert']['embeddings']['position_embeddings']
        w_pe_1 = weights['model_1']['bert']['embeddings']['position_embeddings']
        # Note: We don't need t_out_pos, as position embeddings are added, not concatenated.
        # We assume the main permutation comes from word_embeddings.
        w_pe_fused, _ = fc_fusion(args = args, keys = keys, t_in = None, w_0 = w_pe_0, w_1 = w_pe_1,
                                    act_0 = acts['model_0']['bert']['embeddings']['position_embeddings'],
                                    act_1 = acts['model_1']['bert']['embeddings']['position_embeddings'],
                                    alpha = alpha, device = device, log = log, last_layer = False, is_embed = True)
        w_fused['bert']['embeddings']['position_embeddings'] = {'weight': w_pe_fused['weight'].detach()}

        # --- (CORRECTION 2: Add missing token_type_embeddings fusion) ---
        if 'token_type_embeddings' in weights['model_0']['bert']['embeddings']:
            log.info(' Fusing token type embeddings')
            w_tte_0 = weights['model_0']['bert']['embeddings']['token_type_embeddings']
            w_tte_1 = weights['model_1']['bert']['embeddings']['token_type_embeddings']
            w_tte_fused, _ = fc_fusion(args = args, keys = keys, t_in = None, w_0 = w_tte_0, w_1 = w_tte_1,
                                        act_0 = acts['model_0']['bert']['embeddings']['token_type_embeddings'],
                                        act_1 = acts['model_1']['bert']['embeddings']['token_type_embeddings'],
                                        alpha = alpha, device = device, log = log, last_layer = False, is_embed = True)
            w_fused['bert']['embeddings']['token_type_embeddings'] = {'weight': w_tte_fused['weight'].detach()}
        else:
            log.info(' Skipping token type embeddings (not found in model_0)')

        # embeddings.LayerNorm -> copy from model_1 or fuse if requested
        if args['fusion'].get('fuse_norm', False):
            log.info(' Fusing embeddings LayerNorm')
            w_ln_fused, t_ln = ln_fusion(args = args, keys = keys, t_in = None, # Input T-map is None, as it operates on the sum
                                         w_0 = weights['model_0']['bert']['embeddings']['LayerNorm'],
                                         w_1 = weights['model_1']['bert']['embeddings']['LayerNorm'], alpha = alpha, device = device)
            w_fused['bert']['embeddings']['LayerNorm'] = w_ln_fused
            # If we fuse LN, its output T-map (t_ln) should be used, but ln_fusion just returns t_in.
            # We will rely on the t_out_embed from word_embeddings as the primary map.
            # If t_ln *was* a new map, we'd set t_out_embed = t_ln here.
        else:
            w_fused['bert']['embeddings']['LayerNorm'] = copy.deepcopy(weights['model_1']['bert']['embeddings']['LayerNorm'])
    else:
        log.info(' Copy Embeddings')
        w_fused['bert']['embeddings'] = copy.deepcopy(weights['model_1']['bert']['embeddings'])

    prev_out_acts = acts['model_1'].get('bert.embeddings', {}).get('data', None) if isinstance(acts.get('model_1', {}), dict) else None
    
    # This will hold the T-map from the previous encoder layer
    t_out = None 

    # fuse encoders
    for i in range(number_of_encoders):
        enc_key = str(i)
        
        # --- (CORRECTION 1: Apply T-Map from embeddings to the first encoder layer) ---
        if i == 0:
            current_t_in = t_out_embed  # Pass the map from embeddings
        else:
            current_t_in = t_out         # Pass the map from the previous encoder
            
        last_layer = (i == number_of_encoders-1) and not args['fusion']['fuse_gen']
        
        w_fused['bert']['encoder']['layer'][enc_key], t_out = encoder_fusion(args = args, keys = keys,
                                                                             w_0 = weights['model_0']['bert']['encoder']['layer'][enc_key],
                                                                             w_1 = weights['model_1']['bert']['encoder']['layer'][enc_key],
                                                                             acts_0 = acts['model_0']['bert']['encoder']['layer'][enc_key] if 'model_0' in acts else None,
                                                                             acts_1 = acts['model_1']['bert']['encoder']['layer'][enc_key] if 'model_1' in acts else None,
                                                                             t_in = current_t_in, # Use the corrected T-map
                                                                             last_layer = last_layer, device = device, enc_key = enc_key,
                                                                             alpha = alpha, log = log, prev_out_acts = prev_out_acts)
        
        # update prev_out_acts if activations exist
        if 'model_1' in acts and enc_key in acts['model_1']['bert']['encoder']['layer']:
            prev_out_acts = acts['model_1']['bert']['encoder']['layer'][enc_key].get('data', prev_out_acts)
            
        # --- (修正: 融合 Pooler 层) ---
    if 'pooler' in weights['model_0']['bert']:
        log.info(' Fusing Pooler')
        # t_out 是来自最后一个编码器层的T-Map
        w_pooler_0 = weights['model_0']['bert']['pooler']['dense']
        w_pooler_1 = weights['model_1']['bert']['pooler']['dense']

        # 激活值 (如果 'type: acts')
        act_pooler_0 = acts.get('model_0', {}).get('bert', {}).get('pooler', {}).get('data', None)
        act_pooler_1 = acts.get('model_1', {}).get('bert', {}).get('pooler', {}).get('data', None)

        w_pooler_fused, t_out_pooler = fc_fusion(args = args, keys = keys, t_in = t_out,
                                                w_0 = w_pooler_0, w_1 = w_pooler_1,
                                                act_0 = act_pooler_0, act_1 = act_pooler_1,
                                                alpha = alpha, device = device, log = log, last_layer=False)

        w_fused['bert']['pooler'] = {'dense': w_pooler_fused}

        # 将 Pooler 的 T-map 传递给分类器
        t_out = t_out_pooler
    else:
        log.info(' Skipping Pooler fusion (not found in model_0)')
        w_fused['bert']['pooler'] = copy.deepcopy(weights['model_1']['bert'].get('pooler', {}))
    # --- (结束修正) ---

    # fuse classifier (BERT classification head usually 'classifier')
    if args['fusion'].get('fuse_gen', False) and 'classifier' in weights['model_0']:
        log.info(' Fusing classifier')
        # The input T-map to the classifier is the output T-map (t_out) from the last encoder layer
        w_fused['classifier'], t_out = fc_fusion(args = args, keys = keys, t_in = t_out,
                                                 w_0 = weights['model_0']['classifier'],
                                                 w_1 = weights['model_1']['classifier'],
                                                 act_0 = acts.get('model_0', {}).get('classifier', None),
                                                 act_1 = acts.get('model_1', {}).get('classifier', None),
                                                 alpha = alpha, device = device, log = log, last_layer=True)
    else:
        log.info(' Skipping classifier fusion or classifier not present')
        w_fused['classifier'] = copy.deepcopy(weights['model_1'].get('classifier', {}))

    return w_fused
import time


def prune(args, model, sess, dataset, save_mask=False, save_mask_ci=False, file_prefix='mask', sparsity=0.9, seed=9, arch='lenet300', data='mnist'):
    print('|========= START PRUNING =========|')
    t_start = time.time()
    batch = dataset.get_next_batch('train', args.batch_size)
    feed_dict = {}
    feed_dict.update({model.inputs[key]: batch[key] for key in ['input', 'label']})
    feed_dict.update({model.compress: True, model.is_train: False, model.pruned: False})
    if save_mask:
        result = sess.run([model.outputs, model.mask, model.sparsity], feed_dict)
        import pickle
        with open('./masks_' + arch + '_' + data + '/' + file_prefix + '_sparsity=' + str(sparsity) + '_seed=' + str(seed), 'wb') as f:
            pickle.dump(result[1], f)
        # import sys
        # sys.exit()
    elif save_mask_ci:
        result = sess.run([model.outputs, model.cis, model.sparsity], feed_dict)
        import pickle
        with open('./ci_masks_' + arch + '_' + data + '/' + file_prefix + '_sparsity=' + str(sparsity) + '_seed=' + str(seed), 'wb') as f:
            pickle.dump(result[1], f)
        # import sys
        # sys.exit()
    else:
        result = sess.run([model.outputs, model.sparsity], feed_dict)
        print('Pruning: {:.3f} global sparsity (t:{:.1f})'.format(result[-1], time.time() - t_start))

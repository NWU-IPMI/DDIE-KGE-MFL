# -*- coding: utf-8 -*-
import argparse
import logging
import os
import torch
import numpy as np
import copy
import util
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertConfig, BertTokenizer
from processor_ddie import DDIEProcessor
from modeling_ddie import BertForSequenceClassification
from load_and_cache_examples_ddie import load_and_cache_examples
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
from metrics_ddie import ddie_compute_metrics
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    # parameters
    parser.add_argument("--data_dir", default="", type=str, required=True,
                        help="The input data dir. Should contain the .tsv files for the task.")
    parser.add_argument("--model_name_or_path", default="", type=str, required=True, help="Path to pre-trained model.")
    parser.add_argument("--output_dir", default="", type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--embedding_path", default="", type=str, required=True,
                        help="The path of knowledge graph embeddings.")
    parser.add_argument("--entity_path", default="", type=str, required=True,
                        help="The path of knowledge graph embeddings.")
    parser.add_argument("--stanza_path", default="./stanza_resources/", type=str, required=False,
                        help="The path of stanza.")
    parser.add_argument("--task_name", default="ddie", type=str,
                        help="The default task name should be ddie.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--num_train_epochs", default=5.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If>0: set total number of training steps to perform. Override num_train_epochs")
    parser.add_argument("--max_seq_length", default=390, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", default=False, action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", default=False, action='store_true', help="Whether to run eval on the test set.")
    parser.add_argument("--evaluate_during_training", default=False, action='store_true',
                        help="Whether to run evaluation during training at each loggin step.")
    parser.add_argument("--do_lower_case", default=True, action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--per_gpu_train_batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps", default=500, type=int, help="Log every X updates steps.")
    parser.add_argument("--save_steps", default=-1, type=int, help="Save checkpoint every X updates steps.")
    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA when available.")
    parser.add_argument("--overwrite_output_dir", action='store_true',
                        help="Overwrite the content of the output directory.")
    parser.add_argument("--overwrite_cache", action='store_true',
                        help="Overwrite the cached training and evaluation sets.")
    parser.add_argument("--seed", default=1, type=int, help="Random seed for initialization.")
    parser.add_argument("--local_rank", default=-1, type=int, help="For distributed training: local_rank")
    parser.add_argument("--parameter_averaging", action='store_true', help="Whether to use parameter averaging.")
    parser.add_argument("--dropout_prob", default=0.1, type=float, help="Dropout probability.")
    # del
    parser.add_argument("--middle_layer_size", default=0, type=int, help="Dimention of middle layer.")
    # For CNN
    parser.add_argument("--use_cnn", default=True, action='store_true', help="Whether to use CNN.")
    parser.add_argument("--conv_window_size", default=[3], type=int, nargs='+', help="List of convolution window size.")
    parser.add_argument("--pos_emb_dim", default=20, type=int, help="Dimention of position embeddings.")
    parser.add_argument("--activation", default='relu', type=str, help="Activation function.")
    # For Test
    parser.add_argument("--pretrained_dir", default="", type=str, help="The path to pre-trained model dir.")

    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), False)
    # Set seed
    util.set_seed(args)

    # Prepare task
    processor = DDIEProcessor()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    config = BertConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                        num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                              do_lower_case=args.do_lower_case)
    ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]
    tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
    model = BertForSequenceClassification(args, config, tokenizer)

    if not args.do_train:
        global_step = 0
        if os.path.exists(args.pretrained_dir):
            model.load_state_dict(torch.load(os.path.join(args.pretrained_dir, 'pytorch_model.bin')))
        else:
            raise ValueError("The pre-trained directory is not exist.")
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    model.to(args.device)
    logger.info("Start to train/evaluation ...")

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        global_step, tr_loss, storage_model = train(args, train_dataset, model, tokenizer)
        if global_step % 1000 == 0:
            temp_result = {}
            if args.do_eval:
                temp_result = evaluate(args, model, tokenizer, prefix=str(global_step))
            tmp_result = dict((k + '_{}'.format(global_step), v) for k, v in temp_result.items())
            temp_result.update(temp_result)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)
        logger.info("Saving model checkpoint to %s", args.output_dir)
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'state_dict'))

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        if args.parameter_averaging:
            storage_model.average_params()
            result = evaluate(args, storage_model, tokenizer, prefix="")
        else:
            result = evaluate(args, model, tokenizer, prefix="")
        result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
        results.update(result)
    return results


def train(args, train_dataset, model, tokenizer):
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
    else:
        raise ValueError("Please set local_rank = 1.")
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    bert_params = []
    not_bert_params = []
    for name, params in model.named_parameters():
        if 'bert' in name:
            bert_params += [name]
        else:
            not_bert_params += [name]

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       not any(nd in n for nd in no_decay) and (n in not_bert_params)],
            "weight_decay": args.weight_decay,
            "lr": 5e-4,
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       not any(nd in n for nd in no_decay) and (n in bert_params)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       any(nd in n for nd in no_decay) and (n in not_bert_params)],
            "weight_decay": 0.0,
            "lr": 5e-4,
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       any(nd in n for nd in no_decay) and (n in bert_params)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)

    if args.parameter_averaging:
        storage_model = copy.deepcopy(model)
        storage_model.zero_init_params()
    else:
        storage_model = None

    # Start to train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    entity_embedding = np.load(args.embedding_path, allow_pickle=True)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0

    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    util.set_seed(args)  # Added here for reproductibility (even between python 2 and 3)

    best_f1 = 0.0

    for epoch, _ in enumerate(train_iterator, start=1):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            drug_a_indices = batch[10]
            drug_b_indices = batch[11]
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'relative_dist1': batch[3],
                      'relative_dist2': batch[4],
                      'all_dep_mask0': batch[5],
                      'all_dep_mask1': batch[6],
                      'all_dep_mask2': batch[7],
                      'all_dep_mask3': batch[8],
                      'labels': batch[9],
                      'drug_a_ids': entity_embedding[drug_a_indices],
                      'drug_b_ids': entity_embedding[drug_b_indices]
                      }
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                if not args.parameter_averaging:
                    scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:

                    #add
                    results = evaluate(args, model, tokenizer, prefix="")
                    if results['microF']>best_f1:
                        logger.info("F1_score up! from %.4f to %.4f" % (best_f1, results['microF']))
                        output_dir = os.path.join(args.output_dir, 'best_model')
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model,
                                                                'module') else model  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)
                        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                        logger.info("Saving model checkpoint to %s", output_dir)
                        best_f1 = results['microF']
                    #add end


                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer, prefix="")
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model,
                                                            'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
            if args.parameter_averaging:
                storage_model.accumulate_params(model)
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

        if args.evaluate_during_training:
            prefix = 'epoch' + str(epoch)
            output_dir = os.path.join(args.output_dir, prefix)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            if args.parameter_averaging:
                storage_model.average_params()
                result = evaluate(args, storage_model, tokenizer, prefix=prefix)
                storage_model.restore_params()
            else:
                results = evaluate(args, model, tokenizer, prefix=prefix)

    with open("./result.txt","a+") as fw:
        fw.write("best_f1:%s"%(best_f1))

    if args.local_rank in [-1, 0]:
        tb_writer.close()
    return global_step, tr_loss / global_step, storage_model


def evaluate(args, model, tokenizer, prefix=""):
    """
        evaluate the model
    :param args:
    :param model:
    :param tokenizer:
    :param prefix:
    :return:
    """
    results = {}
    for eval_task, eval_output_dir in zip((args.task_name,), (args.output_dir,)):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        if args.local_rank == -1:
            eval_sampler = SequentialSampler(eval_dataset)
        else:
            raise ValueError("The parameter of local_rank should be -1")
        eval_dataloader = DataLoader(eval_dataset,sampler=eval_sampler,batch_size=args.eval_batch_size)
        entity_embedding = np.load(args.embedding_path,allow_pickle=True)

        # Evaluation
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader,desc="Evaluating"):
            model.eval()

            drug_a_indices = batch[10]
            drug_b_indices = batch[11]
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2],
                    'relative_dist1': batch[3],
                    'relative_dist2': batch[4],
                    'all_dep_mask0': batch[5],
                    'all_dep_mask1': batch[6],
                    'all_dep_mask2': batch[7],
                    'all_dep_mask3': batch[8],
                    'labels': batch[9],
                    'drug_a_ids': entity_embedding[drug_a_indices],
                    'drug_b_ids': entity_embedding[drug_b_indices],
                }
                outputs = model(**inputs)
                tmp_eval_loss,logits = outputs[:2]
                eval_loss+=tmp_eval_loss.mean().item()
            nb_eval_steps+=1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds,logits.detach().cpu().numpy(),axis=0)
                out_label_ids = np.append(out_label_ids,inputs['labels'].detach().cpu().numpy(),axis=0)
        np.save(os.path.join(args.output_dir,'preds'),preds)
        np.save(os.path.join(args.output_dir,'labels'),out_label_ids)
        eval_loss = eval_loss/nb_eval_steps
        preds = np.argmax(preds,axis=1)
        result = ddie_compute_metrics(eval_task,preds,out_label_ids)
        results.update(result)
        output_eval_file = os.path.join(eval_output_dir,prefix,"eval_results.txt")
        with open(output_eval_file,"w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
        return results

if __name__ == "__main__":
    main()

import numpy as np
import tensorflow as tf

from ncinet.config_meta import PredictIngestConfig, EvalConfig, SessionConfig
from typing import Tuple, Mapping


def serialize_model(config, export_dir):
    # type: (SessionConfig, str) -> None
    """Serializes a trained model to disk using the SavedModel format."""
    from ncinet.model import NciKeys

    # Build the computational graph
    graph = tf.Graph()
    logits = config.logits_network_gen(graph, config.model_config, eval_net=True)
    input_tensor = graph.get_tensor_by_name('prints:0')

    # Properly rescale output logits
    if config.xent_type == 'softmax':
        output = tf.nn.softmax(logits)
    elif config.xent_type == 'sigmoid':
        output = tf.sigmoid(logits)
    else:
        raise ValueError

    def load_trained(training_saver, eval_config, session):
        # type: (tf.Saver, EvalConfig, tf.Session) -> int
        """Restores variables from a checkpoint file."""
        checkpoint = tf.train.get_checkpoint_state(eval_config.train_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            # extract global_step from checkpoint filename
            global_step = checkpoint.model_checkpoint_path.split('/')[-1].split('-')[-1]
            # Restores from checkpoint
            with tf.variable_scope(tf.get_variable_scope()):
                training_saver.restore(session, checkpoint.model_checkpoint_path)
        else:
            print('No checkpoint file found')
            raise RuntimeError

        return int(global_step)

    # Add a saver to restore the variable weights
    if config.model_config.is_autoencoder:
        model_ops = graph.get_collection(NciKeys.AE_ENCODER_VARIABLES) \
                    + graph.get_collection(NciKeys.AE_DECODER_VARIABLES)
    else:
        model_ops = graph.get_collection(NciKeys.AE_ENCODER_VARIABLES) \
                    + graph.get_collection(NciKeys.INF_VARIABLES)

    saver = tf.train.Saver(model_ops)

    # Make a session and load variables
    with tf.Session(graph=graph) as sess:
        # Restore the variables
        trained_step = load_trained(saver, config.eval_config, sess)
        print("Loaded variables from training step {}".format(trained_step))

        # Check that the graph is ready
        uninitialized_ops = sess.run(tf.report_uninitialized_variables())
        if len(uninitialized_ops) != 0:
            raise RuntimeError("The following ops are not initialized: {!r}".format(uninitialized_ops))

        # Clear the export directory
        if tf.gfile.Exists(export_dir):
            tf.gfile.DeleteRecursively(export_dir)

        # Save the model
        builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

        graph_inputs = {'prints': tf.saved_model.utils.build_tensor_info(input_tensor)}
        graph_outputs = {'logits': tf.saved_model.utils.build_tensor_info(output)}
        model_sig = tf.saved_model.signature_def_utils.build_signature_def(inputs=graph_inputs,
                                                                           outputs=graph_outputs)

        builder.add_meta_graph_and_variables(sess,
                                             ['ncinet'],
                                             signature_def_map={'ncinet': model_sig})

        builder.save()
        print("Saved computational graph.")


def load_model(model_dir):
    # type: (str) -> Tuple[tf.Session, str, str]
    """Initialize a session from a serialized model.

    Parameters
    ----------
    model_dir: path
        Directory containing serialized model.

    Returns
    -------
    Tuple of (sess, input_name, ouput_name), where `sess` is the restored
    session, `input_name` is the name of the input tensor, and `output_name`
    is the name of the output logits tensor.
    """
    sess = tf.Session(graph=tf.Graph())

    # Restore session from file
    with sess.as_default():
        with sess.graph.as_default():
            meta_graph_def = tf.saved_model.loader.load(sess, ['ncinet'], model_dir)

    # Retrieve input and output tensors from the MetaGraphDef
    signature_def = meta_graph_def.signature_def['ncinet']
    input_name = signature_def.inputs['prints'].name
    output_name = signature_def.outputs['logits'].name

    return sess, input_name, output_name


def generate_predictions(model_path, data_config):
    # type: (str, PredictIngestConfig) -> Mapping[str, np.ndarray]
    """Predict results based on a trained model and input data.

    Parameters
    ----------
    model_path: path
        Path to serialized model.
    data_config: PredictIngestConfig
        Configuration for retrieving prediction data.

    Returns
    -------
    A dictionary from array names to ndarrays. All arrays are in the same order.
    """
    from ncinet.ncinet_input import predict_inputs

    # Retrieve input data
    batch_gen = predict_inputs(data_config)

    # Reconstruct model
    sess, input_name, output_name = load_model(model_path)
    result_tensor = sess.graph.get_tensor_by_name(output_name)

    # Accumulator for predicted results
    output_data = []

    # Do predictions
    for name_batch, print_batch in batch_gen:
        batch_result = sess.run([result_tensor], feed_dict={input_name: list(print_batch)})
        output_data.append((name_batch, batch_result[0]))

    sess.close()

    # Collate data
    results = map(np.concatenate, zip(*output_data))
    return dict(zip(('names', 'logits'), list(results)))

# TieredGQN
This repo is a Keras implementation of the Generative Query Network first described in 'Neural Scene Representation and Rendering' by Eslami and Rezende (2018) at DeepMind.

Datasets can be found at: https://github.com/deepmind/gqn-datasets

# Inspiration
This summer, a DeepMind paper called “Neural scene representation and rendering” caught my attention.  On first glance, I was impressed at how their model was able to construct discrete 3d spacial representations of its environment without any human labeled training data.  However, what really grabbed me was the ability of GQN’s learned representation from rendering new environments to serve as a shortcut in the training of a grasping task of a robotic arm.  As mentioned, they are able to achieve “convergence-level performance with approximately one-fourth as many interactions with the environment as a standard method using raw-pixels”.  It occurred to me that the ability to add generator/decoder networks that take the state representation as input was the most elegant implementation of transfer learning I had seen.  The ability to add new decoder networks to a preexisting encoder function and state representation seems exciting in its modular and flexible nature.

I realized the next logical step was to test the efficacy of this potential for transfer learning as the number of tasks that were end-to-end trained was increased.  For example, would end-to-end training of robotic arm grasping drastically reduce the necessary training steps for the next task?  And if so, would this effect compound and increase transfer learning exponentially?  Conversely, would this sacrifice quality of another decoder network?  Would training across N tasks converge to one set of encoder weights?

My hypothesis is that to enable GQN for modular, extensible transfer learning, you are better off creating higher level state representations than increasing the number of tasks and hoping the state will generalize and/or increasing the size of the layer 1 state representation.

To be more concrete, I plan to perform the same baseline end-to-end training of predictive rendering as they do in this paper as well as end-to-end training of a varying number of additional tasks in order to observe the change in performance of each task over varying levels of extensibility.  This should demonstrate (1) the transfer learning potential of multiple end-to-end trained decoder networks scales across the number tasks assigned to it.

Next, I plan to compare this generalizing performance across a varying range of state representation sizes.  By increasing the size of the state representation, we could argue that more information per task could be encoded, in theory offering generalizability across a greater number of tasks.  Whether this does improve performance or not, we can argue (2) that this is not extensible since changing the state representation encoding size on a trained GQN requires a new encoder with different output dimensions, as well as new decoders with new input dimensions.

Lastly, I want to test the performance of a modified GQN with an additional, “layer 2” state representation on the same tasks as the initial network.  Adding tasks to read from a GQN’s 3d state representation may involve redundantly learning cognitive abilities that would be useful for those individual tasks.  For example, using the 3d state representation as input to a decoder that, for example, must perform a robotic arm grasping task and to another that, say, must use that arm to strike a given pose, involves the redundant learning of how to move and orient that arm in space.  By testing this model, I hope to demonstrate that (3) to minimize the number of samples needed to learn new tasks as well as offer a extensible model, we should instead use GQN’s state representation as input to a higher level state representation that similar tasks such as robotic arm movement would use as input.

## Implementation
This implementation takes a specified number of contexts (images plus their corresponding location), a query (asking the model to render the view from a new location), and a target image (the rendering from that new location).

The encoder network models the "tower" architecture described in the paper.

## Use
#### Local
The GQN model is made up of an Encoder and a Decoder network.  You can test and train them individually using the train_encoder.py and train_decoder.py scripts, respectively.

To train the entirety of the GQN, use the train_gqn.py script.

#### GCloud ML Engine
You can also train the model on Google Cloud running cloud_train.sh.  Note that you will need to change the job name each time you create a new job on ML Engine.


## Contributing
If you'd like to contribute or talk about the model, you can email me or submit a PR.

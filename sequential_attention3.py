# Repeated recognition via accumulate-to-threshold and resetting
#
# In this model, we have a memory that has been pre-initialized to know
# six words (A, B, C, D, E, and F).  If we present these words via the
# "stim" population, then it will start increasing the value of the
# "evidence" population.  If you present a stimulus that it does not know
# (such as G, H, or nothing at all), then it will decrease the value of
# the "evidence" population.
#
# This version introduces an environment that can present one of two
# words, depending on the attention signal.  If attention is W1, then the
# model receives the first word as input, and if it is W2, it receives
# the second word.  The model uses basal ganglia rules to attend to the
# first item, and classify it using the accumulator.  If it is not recognized,
# it will send NO to the motor system.  If it is recognized, then it will
# change attention to W2, reset the accumulator, and classify the next word.
# If that word is recognized, it will output YES, and if it is not recognized
# it will output NO.
#
# You can change what words are presented by changing the "words" variable.

import nengo
import nengo.spa as spa
import numpy as np

D = 16

# define the known words
mem_vocab = spa.Vocabulary(D)
mem_vocab.parse('A+B+C+D+E+F')

# we'll use this to reset the accumulated evidence to zero
reset_vocab = spa.Vocabulary(D)
reset_vocab.parse('EVIDENCE')

# here are the valid motor responses
motor_vocab = spa.Vocabulary(D)
motor_vocab.parse('YES+NO+WAIT')

# and the valid attention signals
attend_vocab = spa.Vocabulary(D)
attend_vocab.parse('W1+W2')

# These are the two words to be presented
words = ['C', 'B']  # both should be recognized; result is YES
#words = ['H', 'B']  # the first is not recognized; result is NO
#words = ['D', 'J']  # the second is not recognized; result is NO

model = spa.SPA()
with model:
    # the current word 
    model.stim = spa.State(D, vocab=mem_vocab)
    
    # the recognition memory  
    model.memory = spa.AssociativeMemory(mem_vocab, 
                                         default_output_key='NONE',
                                         threshold=0.3)
    
    
    nengo.Connection(model.stim.output, model.memory.input)
    
    # this stores the accumulated evidence for or against recognition 
    model.evidence = spa.State(1, feedback=1, feedback_synapse=0.1)
    
    # this scaling factor controls how quickly we accumulate evidence 
    evidence_scale = 0.3
    nengo.Connection(model.memory.am.ensembles[-1], model.evidence.input, transform=-evidence_scale)
    nengo.Connection(model.memory.am.elem_output, model.evidence.input,
                     transform=evidence_scale * np.ones((1,model.memory.am.elem_output.size_out)))

    # we also need a method for resetting the evidence population.  If this
    #  associative memory is set to EVIDENCE then it should reset the
    #  model.evidence population to zero                   
    model.reset = spa.AssociativeMemory(reset_vocab)
    nengo.Connection(model.reset.am.elem_output[0], model.evidence.all_ensembles[0].neurons, 
                     transform=np.ones((model.evidence.all_ensembles[0].n_neurons, 1))*-10, 
                     synapse=0.1)
                     
    # the motor response for producing outputs
    model.motor = spa.State(D, vocab=motor_vocab)
    
    
    # for storing the current attention (W1 or W2)
    model.attention = spa.State(D, vocab=attend_vocab)
    
    # changes the input to model.stim depending on the state of our attention
    def env_node(t, x):
        attn = attend_vocab.dot(x)
        i = np.argmax(attn)
        if attn[i] < 0.5:
            return mem_vocab.parse('0').v
        else:
            return mem_vocab.parse(words[i]).v
    env = nengo.Node(env_node, size_in=D)
    nengo.Connection(model.attention.output, env)
    nengo.Connection(env, model.stim.input)
    
    # action rules for doing the two stages of the task
    #  Note that these all use 0.8 as the threshold for the accumulation
    actions = spa.Actions(
        'dot(attention, W1) - evidence - 0.8 --> motor=NO, attention=W1',
        'dot(attention, W1) + evidence - 0.8 --> attention=W2, reset=EVIDENCE',
        'dot(attention, W1) --> attention=W1',
        'dot(attention, W2) - evidence - 0.8 --> motor=NO, attention=W2',
        'dot(attention, W2) + evidence - 0.8 --> motor=YES, attention=W2',
        'dot(attention, W2) --> attention=W2',
        )
    model.bg = spa.BasalGanglia(actions)
    model.thal = spa.Thalamus(model.bg)
    
    # initialize the model by starting it attending to W1
    model.input = spa.Input(attention = lambda t: 'W1' if t<0.1 else '0')
    
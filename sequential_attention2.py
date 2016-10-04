# Repeated recognition via accumulate-to-threshold and resetting
#
# In this model, we have a memory that has been pre-initialized to know
# six words (A, B, C, D, E, and F).  If we present these words via the
# "stim" population, then it will start increasing the value of the
# "evidence" population.  If you present a stimulus that it does not know
# (such as G, H, or nothing at all), then it will decrease the value of
# the "evidence" population.
#
# We add a basal ganglia and thalamus to thre previous version of the model.
# These implement the detection of a threshold, and produce an output response
# when the threshold is reached.  It also resets the "evidence" accumulator
# when it produces a new output.
#
# If you change the input "stim", the model should output "YES" for recognized
# inputs (A, B, C, D, E, and F) and "NO" for anything else.

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
    
    # implementing the threshold via the basal ganglia
    actions = spa.Actions(
        'evidence --> motor=YES, reset=EVIDENCE',
        '-evidence --> motor=NO, reset=EVIDENCE',
        '0.8 --> motor=WAIT',  # the 0.8 acts as the threshold parameter
        )
    model.bg = spa.BasalGanglia(actions)
    model.thal = spa.Thalamus(model.bg)
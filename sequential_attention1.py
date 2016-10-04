# Recognition by accumulating evidence (plus resetting)
#
# In this model, we have a memory that has been pre-initialized to know
# six words (A, B, C, D, E, and F).  If we present these words via the
# "stim" population, then it will start increasing the value of the
# "evidence" population.  If you present a stimulus that it does not know
# (such as G, H, or nothing at all), then it will decrease the value of
# the "evidence" population.
#
# If we are to use this accumulator more than once, we need a way to reset
# it.  Here, we do this by adding a "reset" population.  If we set "reset_stim"
# to "EVIDENCE", then it will reset the "evidence" population to zero by
# inhibiting the neural activity.


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
    evidence_scale = 0.1
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
                     
    model.reset_stim = spa.State(D, vocab=reset_vocab)
    nengo.Connection(model.reset_stim.output, model.reset.input)
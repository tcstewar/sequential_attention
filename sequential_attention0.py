# Recognition by accumulating evidence
#
# In this model, we have a memory that has been pre-initialized to know
# six words (A, B, C, D, E, and F).  If we present these words via the
# "stim" population, then it will start increasing the value of the
# "evidence" population.  If you present a stimulus that it does not know
# (such as G, H, or nothing at all), then it will decrease the value of
# the "evidence" population.
#
# This system can be used as the basis of an accumulate-to-threshold model
# of recognition, as will be shown in the next examples.


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
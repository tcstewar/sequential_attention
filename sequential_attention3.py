import nengo
import nengo.spa as spa
import numpy as np

D = 16

mem_vocab = spa.Vocabulary(D)
mem_vocab.parse('A+B+C+D+E+F')

reset_vocab = spa.Vocabulary(D)
reset_vocab.parse('EVIDENCE')

motor_vocab = spa.Vocabulary(D)
motor_vocab.parse('YES+NO+WAIT')

attend_vocab = spa.Vocabulary(D)
attend_vocab.parse('W1+W2')

model = spa.SPA()
with model:
    model.stim = spa.State(D, vocab=mem_vocab)
    
    model.memory = spa.AssociativeMemory(mem_vocab, 
                                         default_output_key='NONE',
                                         threshold=0.3)
    
    
    nengo.Connection(model.stim.output, model.memory.input)
    
    
    model.evidence = spa.State(1, feedback=1, feedback_synapse=0.1)
    
    evidence_scale = 0.3
    nengo.Connection(model.memory.am.ensembles[-1], model.evidence.input, transform=-evidence_scale)
    nengo.Connection(model.memory.am.elem_output, model.evidence.input,
                     transform=evidence_scale * np.ones((1,model.memory.am.elem_output.size_out)))
                     
    model.reset = spa.AssociativeMemory(reset_vocab)
    nengo.Connection(model.reset.am.elem_output[0], model.evidence.all_ensembles[0].neurons, 
                     transform=np.ones((model.evidence.all_ensembles[0].n_neurons, 1))*-10, 
                     synapse=0.1)
                     

    
    model.motor = spa.State(D, vocab=motor_vocab)
    
    
    model.attention = spa.State(D, vocab=attend_vocab)
    
    words = ['C', 'B']
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
    
    model.input = spa.Input(attention = lambda t: 'W1' if t<0.1 else '0')
    
from __future__ import print_function

from vowpalwabbit import pyvw


def my_predict(vw, ex):
    pp = 0.
    for f,v in ex.iter_features():
        pp += vw.get_weight(f) * v
    return pp

def ensure_close(a,b,eps=1e-6):
    if abs(a-b) > eps:
        raise Exception("test failed: expected " + str(a) + " and " + str(b) + " to be " + str(eps) + "-close, but they differ by " + str(abs(a-b)))

vw=pyvw.vw('--quiet')
vw.learn("1 |i 1")
ex = vw.example("1 |i 2 |y c")

ex.learn() ; ex.learn()

##################

print('my_predict()',ex.get_updated_prediction())
'''
###############################################################################
vw = pyvw.vw("--quiet")
vw.learn("1 |x a b")

###############################################################################
print('# do some stuff with a read example:')
ex = vw.example("1 |x a b |y c")
ex.learn() ; ex.learn() ; ex.learn() ; ex.learn()

updated_pred = ex.get_updated_prediction()
print('current partial prediction =', updated_pred)

# compute our own prediction
print('        my view of example =', str([(f,v,vw.get_weight(f)) for f,v in ex.iter_features()]))
my_pred = my_predict(vw, ex)
print('     my partial prediction =', my_pred)
ensure_close(updated_pred, my_pred)
print('')
ex.finish()

###############################################################################
print('# make our own example from scratch')
ex = vw.example()
ex.set_label_string("0")
ex.push_features('x', ['a', 'b'])
ex.push_features('y', [('c', 1.)])
ex.setup_example()

print('        my view of example =', str([(f,v,vw.get_weight(f)) for f,v in ex.iter_features()]))
my_pred2 = my_predict(vw, ex)
print('     my partial prediction =', my_pred2)
ensure_close(my_pred, my_pred2)

ex.learn() ; ex.learn() ; ex.learn() ; ex.learn()
print('  final partial prediction =', ex.get_updated_prediction())
ensure_close(ex.get_updated_prediction(), my_predict(vw,ex))
print('')
ex.finish()

'''
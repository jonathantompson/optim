----------------------------------------------------------------------
-- An implementation of RMSPROP
-- As far as I can tell, this method is still unpublished but is from G. Hinton
-- and T. Tieleman, and was presented in Hinton's course on coursera (lecture 
-- 6e).  It's basically a mini-batch approximation of rprop on top of sgd.  It
-- has very obvious parallels to adagrad but has a simpler formation.  It works
-- well with nesterov momentum, but not so well with standard momentum.
--
-- ARGS:
-- opfunc : a function that takes a single input (X), the point of 
--          evaluation, and returns f(X) and df/dX
-- x      : the initial point
-- state  : a table describing the state of the optimizer; after each
--          call the state is modified
--   state.learningRate      : learning rate
--   state.learningRateDecay : learning rate decay
--   state.weightDecay       : weight decay
--   state.momentum          : momentum
--   state.dampening         : dampening for momentum
--   state.nesterov          : enables Nesterov momentum
--   state.learningRates     : vector of individual learning rates
--   state.filterWeight      : moving average filter coefficient for meansq term
--   state.maxgain           : maximum rms value (for stability)
--   state.mingain           : minimum rms value (for stability)
--
-- RETURN:
-- x     : the new x vector
-- f(x)  : the function, evaluated before the update
--
-- (Clement Farabet, 2012)
--
function optim.rmsprop(opfunc, x, config, state)
   -- (0) get/update state
   local config = config or {}
   local state = state or config
   local lr = config.learningRate or 1e-3
   local lrd = config.learningRateDecay or 0
   local wd = config.weightDecay or 0
   local mom = config.momentum or 0
   local damp = config.dampening or mom
   local nesterov = config.nesterov or false
   local lrs = config.learningRates
   state.evalCounter = state.evalCounter or 0
   local nevals = state.evalCounter
   assert(not nesterov or (mom > 0 and damp == 0), "Nesterov momentum requires a momentum and zero dampening")

   local fw = config.filterWeight or 0.9
   local maxgain = config.maxgain or 100.0
   local mingain = config.mingain or 1e-6

   -- (1) evaluate f(x) and df/dx
   local fx,dfdx = opfunc(x)

   -- initialize mean squared error to the first gradient 
   state.ms = state.ms or math.pow(dfdx:norm(2), 2)

   -- (2) weight decay
   if wd ~= 0 then
      dfdx:add(wd, x)
   end

   -- (3) apply momentum
   if mom ~= 0 then
      if not state.dfdx then
         state.dfdx = torch.Tensor():typeAs(dfdx):resizeAs(dfdx):copy(dfdx)
      else
         state.dfdx:mul(mom):add(1-damp, dfdx)
      end
      if nesterov then
         dfdx:add(mom, state.dfdx)
      else
         dfdx = state.dfdx
      end
   end

   -- (4) Add the recent gradient magnitude squared to the moving average
   -- From Hinton's Lecture 6e:
   -- MeanSquare(w,t) = 0.9*MeanSquare(w,t-1) + 0.1*(de/dw)^2
   local grad_mag_sq = math.pow(dfdx:norm(2), 2)
   state.ms = fw * state.ms + (1-fw) * grad_mag_sq
   state.ms = math.min(state.ms, maxgain)
   state.ms = math.max(state.ms, mingain)

   -- (5) learning rate decay (annealing)
   local clr = lr / (1 + nevals*lrd)
   
   -- (6) parameter update with single or individual learning rates
   if lrs then
      if not state.deltaParameters then
         state.deltaParameters = torch.Tensor():typeAs(x):resizeAs(dfdx)
      end
      state.deltaParameters:copy(lrs):cmul(dfdx)
      x:add(-clr/math.sqrt(state.ms), state.deltaParameters)
   else
      x:add(-clr/math.sqrt(state.ms), dfdx)
   end

   -- (7) update evaluation counter
   state.evalCounter = state.evalCounter + 1

   -- return x*, f(x) before optimization
   return x,{fx}
end

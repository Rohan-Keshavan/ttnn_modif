def function_under_test():
    x = 27
    x = x - 1


from tracy import Profiler

profiler = Profiler()

profiler.enable()
function_under_test()
profiler.disable()

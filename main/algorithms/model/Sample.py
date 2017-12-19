class Sample:
    parameters = []
    sampleClass = None

    def __init__(self, p_array, s_class):
        self.parameters = p_array
        self.sampleClass = s_class

    def __eq__(self, other):
        return (self.parameters == other.parameters) and (self.sampleClass == other.sampleClass)

    def __repr__(self):
        return "[P:"+str(self.parameters)+",C:"+str(self.sampleClass)+"]"

    def same_type(self,other):
        if not self.same_class_type(other):
            print("Mismatching class types. ["+str(type(self.sampleClass))+", "+str(type(other.sampleClass))+"]")
            return False
        elif not self. same_param_len(other):
            print("Mismatching number of parameters. ["+str(len(self.parameters))+", "+str(len(other.parameters))+"]")
            return False
        return True

    def same_class_type(self, other):
        return isinstance(self.sampleClass, type(other.sampleClass))

    def same_param_len(self, other):
        return len(self.parameters) == len(other.parameters)




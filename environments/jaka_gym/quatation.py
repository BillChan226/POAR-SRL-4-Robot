import math
class Quaternion:
    def __init__(self, s, x, y, z):
        """构造函数"""
        self.s = s
        self.x = x
        self.y = y
        self.z = z
        self.vector = [x, y, z]
        self.all = [s, x, y, z]

    def __str__(self):
        """输出操作重载"""
        op = [" ", "i ", "j ", "k"]
        q = self.all.copy()
        result = ""
        for i in range(4):
            if q[i] < -1e-8 or q[i] > 1e-8:
                result = result + str(round(q[i], 4)) + op[i]
        if result == "":
            return "0"
        else:
            return result
 
    def __add__(self, quater):
        """加法运算符重载"""
        q = self.all.copy()
        for i in range(4):
            q[i] += quater.all[i]
        return Quaternion(q[0], q[1], q[2], q[3])
 
    def __sub__(self, quater):
        """减法运算符重载"""
        q = self.all.copy()
        for i in range(4):
            q[i] -= quater.all[i]
        return Quaternion(q[0], q[1], q[2], q[3])
 
    def __mul__(self, quater):
        """乘法运算符重载"""
        q = self.all.copy()
        p = quater.all.copy()
        s = q[0]*p[0] - q[1]*p[1] - q[2]*p[2] - q[3]*p[3]
        x = q[0]*p[1] + q[1]*p[0] + q[2]*p[3] - q[3]*p[2]
        y = q[0]*p[2] - q[1]*p[3] + q[2]*p[0] + q[3]*p[1]
        z = q[0]*p[3] + q[1]*p[2] - q[2]*p[1] + q[3]*p[0]
        return Quaternion(s, x, y, z)
 
    def divide(self, quaternion):
        """右除"""
        result = self * quaternion.inverse()
        return result
 
    def modpow(self):
        """模的平方"""
        q = self.all.copy()
        result = q[0]
        for i in range(1, 4):
            result += q[i] ** 2
        return result
 
    def mod(self):
        """求模"""
        return pow(self.modpow(), 1/2)
 
    def conj(self):
        """转置"""
        q = self.all.copy()
        for i in range(1, 4):
            q[i] = -q[i]
        return Quaternion(q[0], q[1], q[2], q[3])
 
    def inverse(self):
        """求逆"""
        q = self.all.copy()
        mod = self.modpow()
        for i in range(4):
            q[i] /= mod
        return Quaternion(q[0], -q[1], -q[2], -q[3])
    def torpy(self):
        q = self.all.copy()
        x = q[1]
        y = q[2]
        z = q[3]
        w = q[0]
        r = math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
        r = r/math.pi*180
        p = math.asin(2*(w*y-x*z))
        p = p/math.pi*180
        y = math.atan2(2*(w*z+y*x), 1-2*(z*z+y*y))
        y = y/math.pi*180
        return r,p,y

import numpy as np

class DataSets():

    def __init__(self):
        super().__init__()

    @staticmethod
    def verify_positive_integer(n):
        if not isinstance(n, int):
            if float(n) == int(n):
                n = int(n)
            else:
                raise ValueError("invalid type(n): {}".format(type(n)))
        if n < 1:
            raise ValueError("invalid n: {}".format(n))
        return n

    def get_annular_dataset(self, n, ndim, r_inner, r_outer):
        n = self.verify_positive_integer(n)
        ndim = self.verify_positive_integer(ndim)
        if ndim > 4:
            raise ValueError("invalid ndim: {}".format(ndim))
        theta = np.linspace(
            0,
            2 * np.pi,
            n // 2)
        noise = np.random.uniform(
            low=0.875,
            high=1.125,
            size=theta.size)
        x_inner = r_inner * np.cos(theta) * noise
        y_inner = r_inner * np.sin(theta) * noise
        x_outer = r_outer * np.cos(theta) * noise
        y_outer = r_outer * np.sin(theta) * noise
        x = np.concatenate((x_inner, x_outer))
        y = np.concatenate((y_inner, y_outer))
        z = np.concatenate((noise, np.exp(noise))) # np.ones(x.size)
        t = np.exp(x)
        if ndim == 1:
            data = np.array([t]).T
        elif ndim == 2:
            data = np.array([x, y]).T
        elif ndim == 3:
            data = np.array([x, y, z]).T
        elif ndim == 4:
            data = np.array([x, y, z, t]).T
        else:
            raise ValueError("invalid ndim: {}".format(ndim))
        return data

    def get_smiley_face(self, n, is_happy=False):
        """
        1 --> face outline
        2 --> left eye
        3 --> right eye
        4 --> mouth
        r ~ radius, t ~ theta, z ~ noise
        """
        ## get number of points per facial feature
        n = self.verify_positive_integer(n)
        n1 = n2 = n3 = n4 = n // 4
        ## get x-coordinate per facial feature
        x1 = x4 = 0
        x2 = 45
        x3 = -1 * x2
        ## get y-coordinate per facial feature
        y1 = 0
        y2 = y3 = 40
        y4 = -70
        ## get radius per facial feature
        r1 = 100
        r2 = r3 = 10
        r4 = 12.5
        ## get angles per eyes/mouth
        left_eye_angle = 3 * np.pi / 4
        right_eye_angle = np.pi / 4
        d_eye_angle = np.pi / 12
        d_mouth_angle = np.pi / 6
        if is_happy:
            face_correction_angle = np.pi
        else:
            face_correction_angle = 0
        left_mouth_angle = left_eye_angle + d_mouth_angle + face_correction_angle
        right_mouth_angle = right_eye_angle - d_mouth_angle + face_correction_angle
        ## get range of angles per facial feature
        t1 = np.linspace(
            0,
            2 * np.pi,
            n1)
        t2 = np.linspace(
            left_eye_angle - d_eye_angle,
            left_eye_angle + d_eye_angle,
            n2)
        t3 = np.linspace(
            right_eye_angle - d_eye_angle,
            right_eye_angle + d_eye_angle,
            n3)
        t4 = np.linspace(
            left_mouth_angle,
            right_mouth_angle,
            n4)
        ## get noise per facial feature
        z1 = np.random.uniform(
            low=0.875,
            high=1.125,
            size=n1)
        z2 = np.random.uniform(
            low=0.925,
            high=1.075,
            size=n2)
        z3 = np.random.uniform(
            low=0.925,
            high=1.075,
            size=n3)
        z4 = np.random.uniform(
            low=0.85,
            high=1.15,
            size=n4)
        ## collect data
        x = np.concatenate((
            r1 * np.cos(t1) * z1 + x1,
            r2 * np.cos(t2) * z2 + x2,
            r3 * np.cos(t3) * z3 + x3,
            r4 * np.cos(t4) * z4 + x4
            ))
        y = np.concatenate((
            r1 * np.sin(t1) * z1 + y1,
            r2 * np.sin(t2) * z2 + y2,
            r3 * np.sin(t3) * z3 + y3,
            r4 * np.sin(t4) * z4 + y4
            ))
        return np.array([x, y]).T


##

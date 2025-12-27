'''
SVAttack
'''

from utils.loadModel import *
from utils.DamperUtils import *
from utils.PerceptualLoss import *
import utils.FileUtils as FileUtil
from utils.ViewUtils import *

from torch.utils.data import DataLoader
from feeder.feeder import *
import argparse
from omegaconf import OmegaConf


class Attacker():
    def __init__(self, args):
        super().__init__()
        self.name = 'SVAttack'
        print(f'Operating parameters {args}')
        self.updateClip = args.updateClip
        self.classWeight = args.classWeight
        self.epochs = args.epochs
        self.model_name = args.model_name
        self.save_path = args.save_path
        self.classifier = nn.DataParallel(getModel(args.model_name))
        self.end_sample_num = args.end_sample_num
        self.class_num = args.class_num
        self.batch_size = args.batch_size
        self.clamp_delta = args.clamp_delta

        data_path = args.data_path
        self.learningRate = 0.01
        label_path = args.label_path
        num_frame_path = args.num_frame_path
        feeder = Feeder(data_path, label_path, num_frame_path)
        self.trainloader = DataLoader(feeder,
                                      batch_size=args.batch_size,
                                      num_workers=4, pin_memory=True, drop_last=True, shuffle=False)
        if args.hookstatus:
            register_hooks(self.classifier, args.hookratio, args.model_name)

    def foolRateCal(self, rlabels, flabels):
        hitIndices = []
        for i in range(0, len(flabels)):
            if flabels[i] != rlabels[i]:
                hitIndices.append(i)
        return len(hitIndices) / len(flabels) * 100

    def getUpdate(self, grads, input):
        return input - grads * self.learningRate

    def distribution_matching_loss(self, pred, flabels):
        pred_mean = torch.mean(pred, dim=0)
        flabels_mean = torch.mean(flabels, dim=0)
        pred_std = torch.std(pred, dim=0)
        flabels_std = torch.std(flabels, dim=0)

        mean_loss = torch.mean((pred_mean - flabels_mean) ** 2)
        std_loss = torch.mean((pred_std - flabels_std) ** 2)

        combined_loss = mean_loss + std_loss
        return combined_loss

    def unspecificAttack(self, labels):
        flabels = np.ones((len(labels), self.class_num))
        flabels = flabels * 1 / self.class_num

        return torch.LongTensor(flabels)

    def attack(self):
        overallFoolRate = 0
        batchTotalNum = 0

        if os.path.exists(self.save_path) == False:
            os.makedirs(self.save_path)
        samples_x_list = []
        frames_list = []
        attck_samples_x_list = []
        attck_samples_y_list = []

        for batchNo, (tx, ty, tn) in enumerate(self.trainloader):
            print(f'batchNo={batchNo}')
            tx = tx.cuda()

            # The attack label is set according to the attack type
            labels = ty

            flabels = torch.ones((len(labels), self.class_num)).cuda()
            flabels = flabels * 1 / self.class_num
            for i in range(len(ty)):
                flabels[i, ty[i]] = 0.001

            # Initialize an empty list to store nonanomalous data
            valid_data = []
            valid_data_y = []
            valid_data_flabels = []


            for i in range(len(tx)):
                if torch.all(tx[i] == 0) == False:
                    valid_data.append(tx[i])
                    valid_data_y.append(ty[i])
                    valid_data_flabels.append(flabels[i])

            for i in range(len(tx) - len(valid_data)):
                valid_data.append(tx[0])
                valid_data_y.append(ty[0])
                valid_data_flabels.append(flabels[0])

            # The non-anomalous data are re-transformed into tensors
            tx = torch.stack(valid_data)
            ty = torch.stack(valid_data_y)
            flabels = torch.stack(valid_data_flabels)

            # 3D to 2d, original 2D samples
            tx_2d = ThreeDimensionsToTwoDimensions(tx)

            adData = tx.clone()
            adData = adData.cuda()
            adData.requires_grad = True
            maxFoolRate = np.NINF
            batchTotalNum += 1

            for ep in range(self.epochs):
                pred = self.classifier(adData)
                predictedLabels = torch.argmax(pred, axis=1)

                classLoss = self.distribution_matching_loss(pred, flabels)

                adData.grad = None
                classLoss.backward(retain_graph=True)
                cgs = adData.grad

                adData_2d = ThreeDimensionsToTwoDimensions(adData)
                # 2D kinematic loss
                projection_loss = computer_perceptual_loss(tx_2d, adData_2d)
                # 3D kinematic loss
                kinematics_loss = computer_perceptual_loss(tx, adData)
                percepLoss = projection_loss + 0.2 * kinematics_loss

                adData.grad = None
                percepLoss.backward(retain_graph=True)
                pgs = adData.grad

                pgsView = pgs.view(pgs.shape[0], -1)
                pgsnorms = torch.norm(pgsView, dim=1) + 1e-18
                pgsView /= pgsnorms[:, np.newaxis]

                if ep % 50 == 0:
                    print(f"Iteration {ep}: Class Loss {classLoss:>9f}, pgs Loss: {percepLoss:>9f}")

                foolRate = self.foolRateCal(ty, predictedLabels)

                if maxFoolRate < foolRate:
                    print('foolRate Improved! Iteration %d, batchNo %d: Class Loss %.9f, pgs: %.9f, Fool rate:%.2f' % (
                        ep, batchNo, classLoss, percepLoss, foolRate))
                    maxFoolRate = foolRate

                if ep == self.epochs - 1:
                    for i in range(len(ty)):
                        if torch.all(tx[i] == 0):
                            print('This raw data is out of order')
                        if torch.sum(tx[i] - adData[i]) != 0:
                            if flabels[i] != predictedLabels[i]:
                                samples_x_list.append(tx[i].detach().clone().cpu())
                                frames_list.append(tn[i].detach().clone().cpu())
                                attck_samples_x_list.append(adData[i].detach().clone().cpu())
                                attck_samples_y_list.append(flabels[i].detach().clone().cpu())
                    break

                cgsView = cgs.view(cgs.shape[0], -1)
                cgsnorms = torch.norm(cgsView, dim=1) + 1e-18
                cgsView /= cgsnorms[:, np.newaxis]

                temp = self.getUpdate(cgs * self.classWeight + pgs * (1 - self.classWeight), adData)

                with torch.no_grad():
                    adData.data = temp.data
                    adData.data = torch.clamp(adData.data, min=tx - self.clamp_delta, max=tx + self.clamp_delta)

            overallFoolRate += maxFoolRate
            print(f"Current fool rate is {overallFoolRate / batchTotalNum}")

            if len(attck_samples_x_list) != 0:
                FileUtil.writer(samples_x_list, frames_list, attck_samples_x_list, attck_samples_y_list, self.save_path)

            if len(attck_samples_x_list) >= self.end_sample_num:
                print('End of attack')
                break

        print(f"Overall fool rate is {overallFoolRate / batchTotalNum}")
        return overallFoolRate / batchTotalNum


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    attacker = Attacker(config)
    attacker.attack()

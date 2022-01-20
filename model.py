import torch
import torch.nn as nn
import torch.nn.functional as F

class DeblurNet(nn.Module):
    def __init__(self):
        super(DeblurNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=15, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.norm1 = nn.BatchNorm2d(num_features=64)
        self.acti1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.norm2 = nn.BatchNorm2d(num_features=64)
        self.acti2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.norm3 = nn.BatchNorm2d(num_features=128)
        self.acti3 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.norm4 = nn.BatchNorm2d(num_features=128)
        self.acti4 = nn.ReLU()

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.norm5 = nn.BatchNorm2d(num_features=256)
        self.acti5 = nn.ReLU()

        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.norm6 = nn.BatchNorm2d(num_features=256)
        self.acti6 = nn.ReLU()

        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.norm7 = nn.BatchNorm2d(num_features=256)
        self.acti7 = nn.ReLU()

        self.conv8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.norm8 = nn.BatchNorm2d(num_features=256)
        self.acti8 = nn.ReLU()

        self.conv9 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.norm9 = nn.BatchNorm2d(num_features=512)
        self.acti9 = nn.ReLU()

        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.norm10 = nn.BatchNorm2d(num_features=512)
        self.acti10 = nn.ReLU()
        
        self.conv11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.norm11 = nn.BatchNorm2d(num_features=512)
        self.acti11 = nn.ReLU()
        
        self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.norm12 = nn.BatchNorm2d(num_features=512)
        self.acti12 = nn.ReLU()

        ##################
        self.conv13 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.norm13 = nn.BatchNorm2d(num_features=256)

        self.acti13 = nn.ReLU()

        self.conv14 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.norm14 = nn.BatchNorm2d(num_features=256)
        self.acti14 = nn.ReLU()

        self.conv15 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.norm15 = nn.BatchNorm2d(num_features=256)
        self.acti15 = nn.ReLU()

        self.conv16 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.norm16 = nn.BatchNorm2d(num_features=256)
        self.acti16 = nn.ReLU()

        ##################
        self.conv17 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.norm17 = nn.BatchNorm2d(num_features=128)

        self.acti17 = nn.ReLU()

        self.conv18 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.norm18 = nn.BatchNorm2d(num_features=128)
        self.acti18 = nn.ReLU()

        self.conv19 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.norm19 = nn.BatchNorm2d(num_features=64)
        self.acti19 = nn.ReLU()

        ##################
        self.conv20 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.norm20 = nn.BatchNorm2d(num_features=64)

        self.acti20 = nn.ReLU()

        self.conv21 = nn.Conv2d(in_channels=64, out_channels=15, kernel_size=3, stride=1, padding=1)
        self.norm21 = nn.BatchNorm2d(num_features=15)
        self.acti21 = nn.ReLU()

        ##################
        self.conv22 = nn.Conv2d(in_channels=15, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.norm22 = nn.BatchNorm2d(num_features=3)

        self.acti22 = nn.Sigmoid()
        

    def forward(self, x):
        ref = x.narrow(1, 7, 3)
        F0 = self.conv1(x)
        F0 = self.norm1(F0)
        F0 = self.acti1(F0)
        
        D1 = self.conv2(F0)
        D1 = self.norm2(D1)
        D1 = self.acti2(D1)

        F1 = self.conv3(D1)
        F1 = self.norm3(F1)
        F1 = self.acti3(F1)
        
        F2 = self.conv4(F1)
        F2 = self.norm4(F2)
        F2 = self.acti4(F2)
        
        D2 = self.conv5(F2)
        D2 = self.norm5(D2)
        D2 = self.acti5(D2)
        
        F3 = self.conv6(D2)
        F3 = self.norm6(F3)
        F3 = self.acti6(F3)
        
        F4 = self.conv7(F3)
        F4 = self.norm7(F4)
        F4 = self.acti7(F4)
        
        F5 = self.conv8(F4)
        F5 = self.norm8(F5)
        F5 = self.acti8(F5)
        
        D3 = self.conv9(F5)
        D3 = self.norm9(D3)
        D3 = self.acti9(D3)
        
        F6 = self.conv10(D3)
        F6 = self.norm10(F6)
        F6 = self.acti10(F6)
        
        F7 = self.conv11(F6)
        F7 = self.norm11(F7)
        F7 = self.acti11(F7)
        
        F8 = self.conv12(F7)
        F8 = self.norm12(F8)
        F8 = self.acti12(F8)
        
        U1 = self.conv13(F8)
        U1 = self.norm13(U1)
        S1 = U1 + F5
        S1 = self.acti13(S1)


        F9 = self.conv14(S1)
        F9 = self.norm14(F9)
        F9 = self.acti14(F9)
        
        F10 = self.conv15(F9)
        F10 = self.norm15(F10)
        F10 = self.acti15(F10)
        
        F11 = self.conv16(F10)
        F11 = self.norm16(F11)
        F11 = self.acti16(F11)
        
        U2 = self.conv17(F11)
        U2 = self.norm17(U2)
        S2 = U2 + F2
        S2 = self.acti17(S2)
        

        F12 = self.conv18(S2)
        F12 = self.norm18(F12)
        F12 = self.acti18(F12)
        
        F13 = self.conv19(F12)
        F13 = self.norm19(F13)
        F13 = self.acti19(F13)
        
        U3 = self.conv20(F13)
        U3 = self.norm20(U3)
        S3 = U3 + F0
        S3 = self.acti20(S3)

        
        F14 = self.conv21(S3)
        F14 = self.norm21(F14)
        F14 = self.acti21(F14)
        

        F15 = self.conv22(F14)
        F15 = self.norm22(F15)
        S4 = F15 + ref
        S4 = self.acti22(S4)

		
        return S4
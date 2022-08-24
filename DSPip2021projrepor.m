
clc
clear
close all
% Inserting reference and distorted image

org_img = 'I04.BMP';
I = imread(org_img);       Id = im2double(I);

% Inserting distorted AWGN image

noisi_img_AWGN01 = 'i04_01_5.bmp';

N_AWGN01 = imread(noisi_img_AWGN01);   Nd_AWGN01 = im2double(N_AWGN01);

% Inserting image - distorted image  by ACN(Additive Color Noise)

noisi_img_ACN = 'i04_02_5.bmp';

N_ACN = imread(noisi_img_ACN);   Nd = im2double(N_ACN);

% loading IN6 (Impulse Noise)

noisi_img_IN6 = 'i04_06_5.bmp';

N_IN6 = imread(noisi_img_IN6);   Nd_IN6 = im2double(N_IN6);

psigtono_IN6 = psnr(I, N_IN6 );

% Extractiong color map and components of RGB reference image to process the distorted image

reref = I(: , :, 1); gref = I(:, :, 2); bref = I(:, :, 3);

% Extractiong color components of AWGN image 
red_AWGN01 = N_AWGN01(: , :, 1); green_AWGN01 = N_AWGN01(:, :, 2); blue_AWGN01 = N_AWGN01(:, :, 3);

% Extractiong color components of ACN image 

red_ACN = N_ACN(: , :, 1); green_ACN = N_ACN(:, :, 2); blue_ACN = N_ACN(:, :, 3);

% Extractiong color map and components of RGB image to process of distorted image

red_IN6 = N_IN6(: , :, 1); green_IN6 = N_IN6(:, :, 2); blue_IN6 = N_IN6(:, :, 3);


% Features of reference Image
s_rgb = size(I);           % size of images

s_com = size(reref);   % size of each comphonets of images

[nrows,ncols] =  size(reref);

img_power  = abs(fftn(I)).^2;                                       % reference image power spectrum

avpower_img = sum(img_power(:))/prod(size(I));      % average power of reference image

Eref_img = fftshift(real(ifftn(img_power)));                 % Energy - autocorrolation of reference image




% Features of distorted image -  AWGN01 distortion type ,
% level 5

Pnoiz_AWGN01 = imsubtract(I,N_AWGN01);           %pure noise for AWGN - level 5

noiz_power_AWGN01 = abs(fftn(Pnoiz_AWGN01)).^2;                                 %  power spectrum of AWGnoise

avpower_noiz__AWGN01 = sum(noiz_power_AWGN01(:))/prod(size(Pnoiz_AWGN01));   % average power of AWGN

E_noiz = fftshift(real(ifftn(noiz_power_AWGN01)));                 % Energy - autocorrolation of AWGN

psigtono_AWGN01 = psnr(N_AWGN01,I);      %peak signal to noise rate of noised img


% Features of distorted ACN image


Pnoiz_ACN = imsubtract(I,N_ACN);                                %pure ACN - level 5



noiz_power_ACN = abs(fftn(Pnoiz_ACN)).^2;                                 %  power spectrum of _ACN

avpower_noiz_ACN = sum(noiz_power_ACN(:))/prod(size(Pnoiz_ACN));   % average power of _ACN

E_noiz_ACN = fftshift(real(ifftn(noiz_power_ACN)));                 % Energy - autocorrolation of _ACN

psigtono_ACN = psnr(N_ACN, I);

% colors comphonets - AWGN

redfft_AWGN01 = fftshift(fft2(real(red_AWGN01)));  grefft_AWGN01 = fftshift(fft2(real(green_AWGN01)));  
blufft_AWGN01 = fftshift(fft2(real(blue_AWGN01)));

figure()
subplot(2,1,1)
imshow(N_AWGN01)
title(['AWGN  Image   ', noisi_img_AWGN01])
subplot(2,1,2)
imhist(N_AWGN01)
title(['Hist. of Distorted Image  ', noisi_img_AWGN01, 'with psnr = ', num2str(psigtono_AWGN01)])



figure()
subplot(3,3,1)
imshow(red_AWGN01)
title('red_AWGN channel')
subplot(3,3,2)
imhist(red_AWGN01)
title('Hist. of red chan')
subplot(3,3,3)
imshow(redfft_AWGN01)
title('freq. dist. of red')

subplot(3,3,4)
imshow(green_AWGN01)
title('green_AWGN channel')
subplot(3,3,5)
imhist(green_AWGN01)
title('Hist. of green chan')
subplot(3,3,6)
imshow(grefft_AWGN01)
title('freq. dist. of green')


subplot(3,3,7)
imshow(blue_AWGN01)
title('blue_AWGN channel')
subplot(3,3,8)
imhist(blue_AWGN01)
title('Hist. of blue chan')
subplot(3,3,9)
imshow(blufft_AWGN01)
title('freq. dist. of blue')


% colors comphonets  - ACN
redfft_ACN = fftshift(fft2(real(red_ACN)));  grefft_ACN = fftshift(fft2(real(green_ACN)));   blufft_ACN= fftshift(fft2(real(blue_ACN)));

figure()
subplot(2,1,1)
imshow(N_ACN)
title(['ACN Image   ', noisi_img_ACN])
subplot(2,1,2)
imhist(N_ACN)
title(['Hist. of ACN Image  ', noisi_img_ACN, 'with psnr = ', num2str(psigtono_ACN)])



figure()
subplot(3,3,1)
imshow(red_ACN)
title('red_ACN channel')
subplot(3,3,2)
imhist(red_ACN)
title('Hist. of red chan')
subplot(3,3,3)
imshow(redfft_ACN)
title('freq. dist. of red')

subplot(3,3,4)
imshow(green_ACN)
title('green_ACN channel')
subplot(3,3,5)
imhist(green_ACN)
title('Hist. of green chan')
subplot(3,3,6)
imshow(grefft_ACN)
title('freq. dist. of green')


subplot(3,3,7)
imshow(blue_ACN)
title('blue channel')
subplot(3,3,8)
imhist(blue_ACN)
title('Hist. of blue chan')
subplot(3,3,9)
imshow(blufft_ACN)
title('freq. dist. of blue')

% colors comphonets  - IN
redfft_IN6 = fftshift(fft2(real(red_IN6)));  grefft_IN6 = fftshift(fft2(real(green_IN6)));   blufft_IN6= fftshift(fft2(real(blue_IN6)));

figure()
subplot(2,1,1)
imshow(N_IN6)
title(['IN Image   ', noisi_img_IN6])
subplot(2,1,2)
imhist(N_IN6)
title(['Hist. of IN Image  ', noisi_img_IN6, 'with psnr = ', num2str(psigtono_IN6)])



figure()
subplot(3,3,1)
imshow(red_IN6)
title('red_IN channel')
subplot(3,3,2)
imhist(red_IN6)
title('Hist. of red chan')
subplot(3,3,3)
imshow(redfft_IN6)
title('freq. dist. of red')

subplot(3,3,4)
imshow(green_IN6)
title('green_IN channel')
subplot(3,3,5)
imhist(green_IN6)
title('Hist. of green chan')
subplot(3,3,6)
imshow(grefft_IN6)
title('freq. dist. of green')


subplot(3,3,7)
imshow(blue_IN6)
title('blue_IN channel')
subplot(3,3,8)
imhist(blue_IN6)
title('Hist. of blue chan')
subplot(3,3,9)
imshow(blufft_IN6)
title('freq. dist. of blue')

% histogram equalization:
%red = histeq(red); green = histeq(green) ; blue = histeq(blue);

% Median filter - AWGN distortion

Rmed_AWGN01= medfilt2(red_AWGN01, [5,5]); Gmed_AWGN01 = medfilt2(green_AWGN01, [5 5]); 
Bmed_AWGN01 = medfilt2(blue_AWGN01, [5 5]);

medfil_img_AWGN01 = cat(3, Rmed_AWGN01, Gmed_AWGN01, Bmed_AWGN01); Psigtonoi2_AWGN01 = psnr(medfil_img_AWGN01, I);


figure()
%title('AWGN distor.')
subplot(2,1,1)
imshow(medfil_img_AWGN01)
title(['medfiltered AWGN Image with psnr =  ',num2str(Psigtonoi2_AWGN01)])
subplot(2,1,2)
imhist(medfil_img_AWGN01)
title('Hist. of medfiltered AWGN Image ')


% Median filter - ACN level IV

Rmed_ACN= medfilt2(red_ACN, [5,5]); Gmed_ACN = medfilt2(green_ACN, [5 5]); Bmed_ACN = medfilt2(blue_ACN, [5 5]);

medfil_img_ACN = cat(3, Rmed_ACN, Gmed_ACN, Bmed_ACN); Psigtonoi2_ACN = psnr(medfil_img_ACN, I);


figure()
subplot(2,1,1)
imshow(medfil_img_ACN)
title(['medfiltered ACN Image with psnr =  ',num2str(Psigtonoi2_ACN)])
subplot(2,1,2)
imhist(medfil_img_ACN)
title('Hist. of medfiltered ACN Image ')

% Median filter - IN level IV

Rmed_IN6= medfilt2(red_IN6, [5,5]); Gmed_IN6 = medfilt2(green_IN6, [5 5]); Bmed_IN6 = medfilt2(blue_IN6, [5 5]);

medfil_img_IN6 = cat(3, Rmed_IN6, Gmed_IN6, Bmed_IN6); Psigtonoi2_IN6 = psnr(medfil_img_IN6, I);


figure()
subplot(2,1,1)
imshow(medfil_img_IN6)
title(['medfiltered IN Image with psnr =  ',num2str(Psigtonoi2_IN6)])
subplot(2,1,2)
imhist(medfil_img_IN6)
title('Hist. of medfiltered IN Image ')


% Mean Filter: 
h_mean = fspecial('average');

meanfil_img1_AWGN01 = imfilter(N_AWGN01, h_mean);              % AWGN distor.

Psigtonoimean_AWGN01 = psnr(meanfil_img1_AWGN01, I);         % AWGN distor.

figure()
subplot(2,1,1)
imshow(meanfil_img1_AWGN01)
title(['meanfiltered AWGN Image with psnr =  ',num2str(Psigtonoimean_AWGN01)])
subplot(2,1,2)
imhist(meanfil_img1_AWGN01)
title('Hist. of meanfiltered AWGN Image ')


meanfil_img1_ACN = imfilter(N_ACN, h_mean);                % ACN

Psigtonoimean_ACN = psnr(meanfil_img1_ACN, I);            % ACN

figure()
subplot(2,1,1)
imshow(meanfil_img1_ACN)
title(['meanfiltered ACN Image with psnr =  ',num2str(Psigtonoimean_ACN)])
subplot(2,1,2)
imhist(meanfil_img1_ACN)
title('Hist. of meanfiltered Image ')

% non uniform mean filter

h_mean2 = (1/16)*([1 2 1; 2 4 2; 1 2 1]);

meanfil_img2_AWGN01 = imfilter(N_AWGN01, h_mean2);           % AWGN distor.

Psigtonoimean2_AWGN01 = psnr(meanfil_img2_AWGN01, I);                         % AWGN distor.

figure()
subplot(2,1,1)
imshow(meanfil_img2_AWGN01)
title(['meanfiltered AWGN Image with psnr =  ',num2str(Psigtonoimean2_AWGN01)])
subplot(2,1,2)
imhist(meanfil_img2_AWGN01)
title('Hist. of meanfiltered AWGN Image ')


meanfil_img2_ACN = imfilter(N_ACN, h_mean2);                  % ACN

Psigtonoimean2_ACN = psnr(meanfil_img2_ACN, I);              % ACN

figure()
subplot(2,1,1)
imshow(meanfil_img2_ACN)
title(['meanfiltered ACN Image with psnr =  ',num2str(Psigtonoimean2_ACN)])
subplot(2,1,2)
imhist(meanfil_img2_ACN)
title('Hist. of meanfiltered Image ')


% Other build-in filters by fspecial command

% Gaussian filters:

h_gau = fspecial('gaussian', 5,2);          % sigma = 2

 % AWGN
gaufil_img_AWGN01 = imfilter(N_AWGN01, h_gau) ;   Psigtonoigau_AWGN01 = psnr(gaufil_img_AWGN01, I); % AWGN

figure()
subplot(2,1,1)
imshow(gaufil_img_AWGN01)
title(['gau-filtered Image with psnr =  ',num2str(Psigtonoigau_AWGN01), 'AWGN'])
subplot(2,1,2)
imhist(gaufil_img_AWGN01)
title('Hist. of gaufiltered Image ', 'AWGN')


%ACN 
gaufil_img_ACN = imfilter(N_ACN, h_gau) ;   Psigtonoigau_ACN = psnr(gaufil_img_ACN, I);              %ACN 

figure()
subplot(2,1,1)
imshow(gaufil_img_ACN)
title(['gau-filtered ACN Image with psnr =  ',num2str(Psigtonoigau_ACN)])
subplot(2,1,2)
imhist(gaufil_img_ACN)
title('Hist. of gaufiltered Image ')

% mean squared error AWGN01
meansqrerror1_AWGN01 = immse(I,N_AWGN01); meansqrerror2_AWGN01 = immse(I,gaufil_img_AWGN01);
figure()
subplot(2,1,1)
imshow(N_AWGN01)
title(['before with mse = ', num2str(meansqrerror1_AWGN01), 'AWGN'])
subplot(2,1,2)
imshow(gaufil_img_AWGN01)
title(['after with mse = ', num2str(meansqrerror2_AWGN01), 'AWGN'])

imwrite(gaufil_img_AWGN01, 'i04AWGN01-rescbygausfiltr.bmp');


% mean squred error - ACN
meansqrerror1_ACN = immse(I,N_ACN); meansqrerror2_ACN = immse(I,meanfil_img1_ACN);
figure()
subplot(2,1,1)
imshow(N_ACN)
title(['ACN - before with mse = ', num2str(meansqrerror1_ACN)])
subplot(2,1,2)
imshow(meanfil_img1_ACN)
title(['after with mse = ', num2str(meansqrerror2_ACN)])
imwrite(meanfil_img1_ACN, 'i04ACN-rescbymeanfiltr.bmp');

% mean squred error - Impulse Noise
meansqrerror1_IN6 = immse(I,N_IN6); meansqrerror2_IN6 = immse(I, medfil_img_IN6);
figure()
subplot(2,1,1)
imshow(N_IN6)
title(['IN - before with mse = ', num2str(meansqrerror1_IN6)])
subplot(2,1,2)
imshow(medfil_img_IN6)
title(['after with mse = ', num2str(meansqrerror2_IN6)])
imwrite(medfil_img_IN6, 'i04IN-rescbymeanfiltr.bmp');

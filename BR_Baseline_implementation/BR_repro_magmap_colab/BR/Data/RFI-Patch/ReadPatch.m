%% read
load('.\s1.mat') % you can change “vh-3.mat” and the Section 1 (Data)
figure;imagesc(fGamm(squeeze(ss(1,:,:)),0.3))
figure;imagesc(fGamm(squeeze(ss(2,:,:)),0.3))
figure;imagesc(fGamm(squeeze(ss(3,:,:)),0.3))
figure;imagesc(fGamm(squeeze(ss(4,:,:)),0.3))
figure;imagesc(fGamm(squeeze(ss(5,:,:)),0.3))
%% print SAR
function gamman = fGamm(Image,Th)
gamman                  = sqrt(abs(Image));
peak                    = max(max(gamman)); 
gamman                  = gamman/peak;
gamman                  = gamman-min(min(gamman));
mu                      = mean(mean(gamman));
gamman                  = min(gamman/mu*Th,1);
end

clc;clear
load mpii_human_pose_v1_u12_1.mat

annolist = RELEASE.annolist;
train_num = find(RELEASE.img_train==1);
path = '/MPI/images/';
fid = fopen('data.txt', 'w');

for i = 1:length(train_num)
    try
        index = train_num(i);
        image = [path annolist(index).image.name];
        annorect = annolist(index).annorect;
        imsize = size(imread(image));
        for j = 1:length(annorect)
            point = annorect(j).annopoints.point;
            objpos = annorect(j).objpos;
            joints = zeros(16,2) - 1;
            for k = 1:length(point)
                id = point(k).id;
                joints(id+1,:) = [point(k).x point(k).y];
                
            end
            joints = reshape(joints', 1, []);
            fprintf(fid, '%s\t%d %d %g %g %s\n', image, imsize(1), imsize(2), objpos.x, objpos.y, num2str(joints));
        end
    catch
        continue
    end
end
fclose(fid);

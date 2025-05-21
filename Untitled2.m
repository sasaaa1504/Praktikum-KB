% Jaringan syaraf Metode Backpropagation
% untuk operasi XOR dengan Mathlab

clear;
X = [0 0; 0 1; 1 0; 1 1];     % Input
T = [0; 1; 1; 0];             % Target
alfa = 0.35;                  % Learning rate
Eps = 1e-10;                  % Toleransi error
MaxEpoh = 1000;               % Maksimum Epoh
[m,n] = size(X);

sel1 = 4;                     % Jumlah neuron pada hidden layer
sel2 = 1;                     % Jumlah neuron pada output layer

% Bobot-bobot input layer ke hidden layer
v = [0.9562   0.7762   0.1623   0.2886; 0.1962   0.6133   0.0311   0.9711];
v0 = [0.7496  0.3796  0.7256  0.1628];

% Bobot-bobot hidden layer ke output layer
w = [0.2280; 0.9585; 0.6799; 0.0550];
w0 = [0.9505 ];

% Posisi epoh yang akan disimpan dalam file
awal = MaxEpoh;   % Sebelum epoh ke
akhir = MaxEpoh;  % Sesudah epoh ke

% Menyimpan kondisi awal di file HasBackprob.m
fb = fopen('HasBackprob.m','w');
fprintf(fb,'X = \n');
for i=1:m,
    fprintf(fb,'%3d %3d\n',X(i,:));
end;
fprintf(fb,'\n');
fprintf(fb,'T = \n');
for i=1:m,
    fprintf(fb,'%3d\n',T(i));
end;
fprintf(fb,'\n');
fprintf(fb,'Jumlah neuron pada input layer  = %3d\n',n);
fprintf(fb,'Jumlah neuron pada hidden layer = %3d\n',sel1);
fprintf(fb,'Jumlah neuron pada output layer = %3d\n',sel2);
fprintf(fb,'Learning rate = %2.4f\n',alfa);
fprintf(fb,'Maksimum Epoh = %5d\n',MaxEpoh);
fprintf(fb,'Target Error = %0.5g\n',Eps);
fprintf(fb,'\n');
fprintf(fb,'Bobot Awal input ke hidden (v):\n');
for g=1:n,
    fprintf(fb,'%8.4f %8.4f %8.4f %8.4f\n',v(g,:));
end;
fprintf(fb,'\n');
fprintf(fb,'Bobot Awal bias ke hidden (b1):\n');
for g=1:sel1,
    fprintf(fb,'%8.4f %8.4f\n',v0(g));
end;
fprintf(fb,'\n');
fprintf(fb,'Bobot Awal hidden ke output (w):\n');
for g=1:sel1,
    fprintf(fb,'%8.4f\n',w(g));
end;
fprintf(fb,'\n');
fprintf(fb,'Bobot Awal bias ke output (b2):\n');
for g=1:sel2,
    fprintf(fb,'%8.4f\n',w0(g));
end;

% Mulai Iterasi
epoh = 0;
tMSE = 1;
while (epoh<MaxEpoh) && (tMSE>Eps),
    epoh=epoh+1;
    
    if (epoh<=awal)||(epoh>akhir),
        fprintf(fb,'\n\nEpoh ke = %3d\n',epoh);
        fprintf(fb,'--------------------\n');
    end;
    E=0;

    % Kerjakan mulai dari data ke-1 sampai ke-n
    for k=1:m,

        % Hitung z, hasil operasi input ke hidden layer
        for i=1:sel1,
            z_in(i) = v0(i);
            for j=1:n,
                z_in(i) = z_in(i) + v(j,i)*X(k,j);
            end;
            z(i) = 1/(1+exp(-1*z_in(i)));   % Fungsi aktivasi
        end;

        % Simpan hasilnya
        if (epoh<=awal)||(epoh>akhir),
            fprintf(fb,'\n');
            fprintf(fb,'Data ke = %1d\n',k);
            fprintf(fb,'\n');
            fprintf(fb,'  o Operasi pada Hidden Layer ---> \n');
            fprintf(fb,'    Perkalian [z_in=b1+jumlah(v*X)] :\n');
            for g=1:sel1,
                fprintf(fb,'%8.4f %8.4f\n',z_in(g));
            end;
            fprintf(fb,'\n');
            fprintf(fb,'    Pengaktifan [z=f(z_in)] :\n');
            for g=1:sel1,
                fprintf(fb,'%8.4f %8.4f\n',z(g));
            end;
        end;
    
    % Hitung y, hasil operasi hidden ke output layer
    for i=1:sel2,
        y_in(i) = w0(i);
        for j=1:sel1,
            y_in(i) = y_in(i) + w(j,i)*z(j);
        end;
        %y(k,i)=1/(1+exp(-1*y_in(i)));   % Fungsi aktivasi sigmoid
        y(k,i)=y_in(i);               % Fungsi aktivasi identitas
    end;

    % simpan hasilnya
    if (epoh<=awal)||(epoh>akhir),
        fprintf(fb,'\n');
        fprintf(fb,'  o Operasi pada Output Layer ---> \n');
        fprintf(fb,'    Perkalian [y_in=b2+jumlah(w*z)] :\n');
        for g=1:sel2,
            fprintf(fb,'%8.4f %8.4f\n', y_in(g));
        end;
        fprintf(fb,'\n');
        fprintf(fb,'    Pengaktifan [y=f(y_in)] :\n');
        for g=1:sel2,
            fprintf(fb,'%8.4f %8.4f\n', y(g));
        end;
    end;

    % Hitung Sum Square Error
    err=T(k)-y(k,i);
    E=E+err*err;

    % Simpan hasilnya
    if (epoh<=awal)||(epoh>akhir),
        fprintf(fb,'\n');
        fprintf(fb,'  o Error = %4.2f (T-y)\n',err);
        fprintf(fb,'  o Jumlah Kuadrat Error = %4.2f (E=E+Error*Error)\n',E);
        fprintf(fb,'  o Informasi error dari output layer:\n');
    end;

    % Hitung perambatan error dari output layer ke hidden layer
    for i=1:sel2,
        %delta(i)= (T(k)-y(k,i))((1/(1+exp(-1*y_in(i))))(1-(1/(1+exp(-1*y_in(i)))))); % Turunan Sigmoid
        delta(i)= (T(k)-y(k,i));   % Turunan Identitas
        dw0(i) = alfa*delta(i);

        % Simpan hasilnya
        if (epoh<=awal)||(epoh>akhir),
            fprintf(fb,'  Delta ke-%1d = %4.2f (Error*f(y_in))\n',i,delta(i));
            fprintf(fb,'    Perubahan Bobot Bias [db2(%1d)] = %6.4f (alfa*Delta)\n',i,dw0(i));
        end;
        for j=1:sel1,
            dw(j,i) = alfa*delta(i)*z(j);
            if (epoh<=awal)||(epoh>akhir),
                fprintf(fb,'  Perubahan Bobot Lapisan [dw(%1d,%1d)] = %6.4f (alfa*Delta*z(%1d,%1d))\n',j,i,dw(j,i),j,i);
            end;
        end;
     end;

    if (epoh<=awal)||(epoh>akhir),
        fprintf(fb,'  o Informasi error dari hidden layer:\n');
    end;

% Hitung perambatan error dari hidden layer ke input layer
    for i=1:sel1,
        delta_in(i)=0;
        for j=1:sel2,
            delta_in(i)= delta_in(i)+delta(j)*w(i,j);
        end;
        delta1(i) = delta_in(i)(1/(1+exp(-1*z_in(i))))(1-(1/(1+exp(-1*z_in(i)))));
        dv0(i) = alfa*delta1(i);

        % simpan hasilnya
        if (epoh<=awal)||(epoh>akhir),
            fprintf(fb,'  Delta_in ke-%1d = %4.2f (Sum(Delta*W))\n',i,delta_in(i));
            fprintf(fb,'  Delta ke-%1d = %4.2f (Delta_in*f(z_in))\n',i,delta1(i));
            fprintf(fb,'    Perubahan Bobot Bias [db1(%1d)] = %6.4f (alfa*Delta)\n',i,dv0(i));
        end;

        for j=1:n,
            dv(j,i) = alfa*delta1(i)*X(k,j);
            if (epoh<=awal)||(epoh>akhir),
                fprintf(fb,'    Perubahan Bobot Input [dv(%1d,%1d)] = %6.4f (alfa*Delta*X(%1d,%1d))\n',j,i,dv(j,i),j,i);
            end;
        end;
    end;
    
% Hitung perubahan bobot
    for i=1:sel2,
        w0(i)=w0(i)+dw0(i);
        for j=1:sel1,
            w(j,i) = w(j,i)+dw(j,i);
        end;
    end;

    for i=1:sel1,
        v0(i)=v0(i)+dv0(i);
        for j=1:n,
            v(j,i) = v(j,i)+dv(j,i);
        end;
    end;

% Simpan bobot akhir
    if (epoh<=awal)||(epoh>akhir),
        fprintf(fb,'\n');
        fprintf(fb,'  o Bobot Akhir input ke hidden (v = v + dv):\n');
        for g=1:n, 
                fprintf(fb,' %8.4f %8.4f %8.4f\n', v(g,:));  % kemungkinan ini harusnya baris per baris
        end;
        fprintf(fb,'\n');
        fprintf(fb,'  o Bobot Akhir bias ke hidden (b1 = b1 + db1):\n');
        for g=1:sel1,
            fprintf(fb,' %8.4f\n', v0(g));
        end;
        fprintf(fb,'\n');
        fprintf(fb,'  o Bobot Akhir hidden ke output (w = w + dw):\n');
        for g=1:sel1,  
            fprintf(fb,' %8.4f\n', w(g,:));
        end;
        fprintf(fb,'\n');
        fprintf(fb,'  o Bobot Akhir bias ke output (b2 = b2 + db2):\n');
        for g=1:sel2,
            fprintf(fb,' %8.4f\n', w0(g));
        end;
    end;

    % Hitung MSE
    MSE(epoh) = E/m;
    tMSE = MSE(epoh);
    if (epoh<=awal)||(epoh>akhir),
        fprintf(fb,'\nMean Square Error (MSE) = %0.5g\n', tMSE);
    end;

    ke(epoh) = epoh;
    plot(ke, MSE, 'linewidth', 2);
    title(['Grafik MSE tiap epoh (Epoh ke- ', int2str(epoh), ' ; MSE=', num2str(MSE(epoh)), ')']);
    xlabel('Epoh'); ylabel('MSE'); grid;
    disp(strcat('Epoh ke-', int2str(epoh), ', MSE = ', num2str(MSE(epoh))));
    pause(0.1);
end;
end;

% Simpan hasil
fprintf(fb,'\n');
fprintf(fb,'Epoh Akhir = %3d\n', epoh);
fprintf(fb,'----------------------------\n');
fprintf(fb,'Bobot Akhir input ke hidden \n');
fprintf(fb,'v = \n');
for g=1:n,
    fprintf(fb,'%8.4f %8.4f %8.4f\n', v(g,:));  % cetak akhir bobot input ke hidden
end;
fprintf(fb, '\n');
fprintf(fb,'Bobot Akhir bias ke hidden \n');
fprintf(fb,'b1 = \n');
for g=1:sel1,
    fprintf(fb,'%8.4f %8.4f\n', v0(g));
end;
fprintf(fb,'\n');
fprintf(fb,'Bobot Akhir hidden ke output \n');
fprintf(fb,'w = \n');
for g=1:sel1,
    fprintf(fb,'%8.4f\n', w(g,:));
end;
fprintf(fb,'\n');
fprintf(fb,'Bobot Akhir bias ke output \n');
fprintf(fb,'b2 = \n');
for g=1:sel2,
    fprintf(fb,'%8.4f %8.4f\n', w0(g));
end;
fprintf(fb,'\n');

% Lakukan pengujian terhadap data training
[bb, cc] = size(X);
for k=1:bb,
    for i=1:sel1,
        z_inc(i) = v0(i);
        for j=1:n,
            z_inc(i) = z_inc(i) + v(j,i)*X(k,j);
        end;
        zc(i) = 1/(1+exp(-1*z_inc(i)));
    end;

    for i=1:sel2,
        y_inc(k,i) = w0(i);
        for j=1:sel1,
            y_inc(k,i) = y_inc(k,i) + w(j,i)*zc(j);
        end;
        %yc(k,i) = 1/(1+exp(-1*y_inc(k,i)));
        yc(k,i)= y_inc(k,i);
    end;
end;

fclose(fb);

plot(ke, MSE, 'linewidth', 2);
title(['Grafik MSE tiap epoh (Epoh ke- ', int2str(epoh), ' ; MSE=', num2str(MSE(epoh)), ')']);
xlabel('Epoh'); ylabel('MSE'); grid;
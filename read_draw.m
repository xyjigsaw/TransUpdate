clc
clear
path = 'result0409fromServer/';
cmp_name = 'transr/';
[new_triples_my, test_num_my, h_mr_my, h_hit_my, t_mr_my, t_hit_my, ave_mr_my, ave_hit_my] = textread([path, cmp_name, 'my_raw.csv'], '%f%f%f%f%f%f%f%f', 'delimiter', ',');
[new_triples_t, test_num_t, h_mr_t, h_hit_t, t_mr_t, t_hit_t, ave_mr_t, ave_hit_t] = textread([path, cmp_name, 'transr_raw.csv'], '%f%f%f%f%f%f%f%f', 'delimiter', ',');

plot(new_triples_my, ave_mr_my, '-*b', new_triples_my , ave_mr_t, '-or');
xlabel('number of new triples')
ylabel('MeanRank');
legend('My', 'TransE');
yyaxis right;

% xlim = ([-2,6]);

plot(new_triples_my, ave_hit_my, '--.g', new_triples_my , ave_hit_t, '--+k');
ylabel('Hits@10');
legend('My-MeanRank', 'TransE-MeanRank', 'My-Hits@10', 'TransE-Hits@10');

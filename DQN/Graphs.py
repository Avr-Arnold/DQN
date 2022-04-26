
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def generate_gammas():
    df95 = pd.read_csv('G95.csv')
    df995 = pd.read_csv('G995.csv')
    best = pd.read_csv('Best.csv')
    df995['gamma'] = '.995'
    best['gamma'] = '.99'
    df = pd.concat((df95, best, df995))
    df.rename(columns={'gamma': 'Gamma', 'Unnamed: 0': 'Episodes', 'rewards': 'Episode Reward', 'last_rewards_mean': 'Rewards Mean'}, inplace=True)
    # df['RMSE'] = RMSE
    # df[''] = lambdas
    sns_plot = sns.lineplot(data=df, x='Episodes', y='Rewards Mean', hue='Gamma').set_title('Gamma')
    plt.show()
    fig = sns_plot.get_figure()
    fig.savefig('Gamma')
    return

def generate_epsilon_decay():
    df94 = pd.read_csv('Dec94R2.csv')
    df96 = pd.read_csv('Dec96R2.csv')
    best = pd.read_csv('Best.csv')


    df = pd.concat((df94, df96, best))
    df.rename(columns={'Unnamed: 0': 'Episodes', 'rewards': 'Episode Reward', 'last_rewards_mean': 'Rewards Mean', 'epsilon_decay': 'Epsilon Decay', 'epsilon_decay_list': 'Epsilon'}, inplace=True)
    # df['RMSE'] = RMSE
    # df[''] = lambdas
    sns_plot = sns.lineplot(data=df, x='Episodes', y='Rewards Mean', hue='Epsilon Decay').set_title('Epsilon Decay')
    plt.show()
    fig = sns_plot.get_figure()
    fig.savefig('Epsilon Decay')
    return


def generate_alpha():
    df0025 = pd.read_csv('A0025.csv')
    df00025 = pd.read_csv('A00025.csv')
    best = pd.read_csv('Best.csv')
    best['alpha'] = '.001'

    df = pd.concat((df0025, best, df00025))
    df.rename(columns={'Unnamed: 0': 'Episodes', 'rewards': 'Episode Reward', 'last_rewards_mean': 'Rewards Mean', 'alpha': 'Alpha'}, inplace=True)
    #err = df['Episode Reward'].values*.5
    #dict =  {'x': df['Rewards Mean'], 'y1':df['Rewards Mean'].values - err, 'y2':df['Rewards Mean'].values + err}
    # df['RMSE'] = RMSE
    # df[''] = lambdas
    sns_plot = sns.lineplot(data=df, x='Episodes', y='Rewards Mean', hue='Alpha').set_title('Alpha')#,err_style='band', err_kws=dict
    sns.despine()
    plt.show()
    fig = sns_plot.get_figure()
    fig.savefig('Alpha')
    return

def generate_training_rewards():
    best = pd.read_csv('Best.csv')
    best.rename(columns={'Unnamed: 0': 'Episodes', 'rewards': 'Episode Reward', 'last_rewards_mean': 'Rewards Mean'}, inplace=True)

    sns_plot = sns.scatterplot(data=best, x='Episodes', y='Episode Reward').set_title(
        'Episode Reward')  # ,err_style='band', err_kws=dict
    #sns.despine()
    plt.show()
    fig = sns_plot.get_figure()
    fig.savefig('Episode Rewards')
    return

def generate_trained_rewards():
    best = pd.read_csv('Trained.csv')
    best.rename(columns={'Unnamed: 0': 'Episodes', 'rewards': 'Episode Reward', 'last_rewards_mean': 'Rewards Mean'}, inplace=True)

    sns_plot = sns.scatterplot(data=best, x='Episodes', y='Episode Reward').set_title('Episode Reward')
    #sns.despine()
    plt.show()
    fig = sns_plot.get_figure()
    fig.savefig('Trained Episode Rewards')
    return

if __name__ == '__main__':
    #generate_gammas()
    #generate_epsilon_decay()
    #generate_alpha()
    #generate_training_rewards()
    generate_trained_rewards()
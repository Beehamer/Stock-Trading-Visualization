
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import style
import pandas as pd
from dateutil import parser

# finance module is no longer part of matplotlib
# see: https://github.com/matplotlib/mpl_finance
from mpl_finance import candlestick_ochl as candlestick

style.use('dark_background')

VOLUME_CHART_HEIGHT = 0.33

UP_COLOR = '#27A59A'
DOWN_COLOR = '#EF534F'
UP_TEXT_COLOR = '#73D3CC'
DOWN_TEXT_COLOR = '#DC2C27'


def date2num(date):
    converter = mdates.strpdate2num('%Y-%m-%d')
    return converter(date)


class StockTradingGraph:
    """A stock trading visualization using matplotlib made to render OpenAI gym environments"""

    def __init__(self, df, training_set_size, title=None):
        self.df = df
        self.net_worths = np.zeros(len(df['Date']))
        self.profit_values = np.zeros(len(df['Date']))

        self.training_set_size = training_set_size


        # Create a figure on screen and set the title
        fig = plt.figure()
        fig.suptitle(title)

        # Loading the s&p 500 data
        self.sp_df = pd.read_csv('./data/^GSPC.csv')
      
        # lags = (parser.parse(df['Date'].iloc[0]) - parser.parse(self.sp_df['Date'].iloc[0])).days
        #mask = (self.sp_df['Date'] == str(df['Date'].iloc[0]))
        #lags = int(self.sp_df.loc[mask])

        lags = self.sp_df.index[self.sp_df['Date'] == df['Date'].iloc[0]].tolist()[0]

        print("Length of  self.sp_df " + str(len(self.sp_df)) + "and lags is" + str(lags))

        self.sp_df = self.sp_df[lags:lags + len(df['Date'])]
        # converting them to the actual profit
        self.sp_array = self.sp_df['Close'].to_numpy()
        print("Length of  self.sp_array " + str(len(self.sp_array)) + " and the lags is:" + str(lags))
        self.initial_sp_stock_num = 10000.0 / self.sp_array[self.training_set_size]
        self.sp_worth = np.full(len(self.sp_array), 10000)


        for i in range(self.training_set_size, len(self.sp_array)):
            self.sp_worth[i] = self.initial_sp_stock_num * self.sp_array[i]

        self.buy_and_hold = np.zeros(len(df['Date']))
        self.buy_and_hold[0] = 10000

        if 'Adjusted_Close' in df:
            self.initial_stock_num_buy_and_hold = 10000.0 / self.df['Adjusted_Close'].iloc[self.training_set_size]
            for i in range(self.training_set_size, len(df['Date'])):
                self.buy_and_hold[i] = self.initial_stock_num_buy_and_hold * df['Adjusted_Close'].iloc[i]
        else:
            self.initial_stock_num_buy_and_hold = 10000.0 / self.df['Close'].iloc[self.training_set_size]
            for i in range(self.training_set_size, len(df['Date'])):
                self.buy_and_hold[i] = self.initial_stock_num_buy_and_hold * df['Close'].iloc[i]

        # Add one more subplot for the profit
        self.profit_ax = plt.subplot2grid((6, 1), (0, 0), rowspan= 2, colspan= 1)
        # Create top subplot for net worth axis
        self.net_worth_ax = plt.subplot2grid(
            (6, 1), (2, 0), rowspan=2, colspan=1, sharex = self.profit_ax)

        # Create bottom subplot for shared price/volume axis
        self.price_ax = plt.subplot2grid(
            (6, 1), (4, 0), rowspan=8, colspan=1, sharex=self.net_worth_ax)

        # Create a new axis for volume which shares its x-axis with price
        self.volume_ax = self.price_ax.twinx()

        # Add padding to make graph easier to view
        plt.subplots_adjust(left=0.11, bottom=0.24,
                            right=0.90, top=0.90, wspace=0.2, hspace=0)

        # Show the graph without blocking the rest of the program
        plt.show(block=False)

    def _render_net_worth(self, current_step, net_worth, step_range, dates):
        # Clear the frame rendered last step
        self.net_worth_ax.clear()

        # Plot net worths
        self.net_worth_ax.plot_date(
            dates, self.net_worths[step_range], '-', label='Net Worth')
        self.net_worth_ax.plot_date(
            dates, self.sp_worth[step_range], '-', label='S&P 500', color = "blue")
        self.net_worth_ax.plot_date(
            dates, self.buy_and_hold[step_range], '-', label='Buy and Hold', color = "yellow")
        # Show legend, which uses the label we defined for the plot above
        self.net_worth_ax.legend()
        legend = self.net_worth_ax.legend(loc=2, ncol=2, prop={'size': 8})
        legend.get_frame().set_alpha(0.4)

        last_date = date2num(self.df['Date'].values[current_step])
        last_net_worth = self.net_worths[current_step]

        # Annotate the current net worth on the net worth graph
        self.net_worth_ax.annotate('{0:.2f}'.format(net_worth), (last_date, last_net_worth),
                                   xytext=(last_date, last_net_worth),
                                   bbox=dict(boxstyle='round',
                                             fc='w', ec='k', lw=1),
                                   color="black",
                                   fontsize="small")

        # Add space above and below min/max net worth
        self.net_worth_ax.set_ylim(
            min(self.net_worths[np.nonzero(self.net_worths)]) / 1.25, max(self.net_worths) * 1.25)


    def _render_profit(self, current_step, net_worth, step_range, dates, starting_net_worth):
        # Clear the frame rendered last step
        self.profit_ax.clear()
        profit = net_worth - starting_net_worth
        # if current_step == 0:
        #     sp_profit = 0
        # else:
        #     sp_profit = (self.sp_array[current_step] - self.sp_array[current_step - 1]) * self.initial_spstock_num
        self.profit_ax.plot_date(
            dates, self.profit_values[step_range], '-', label='Profit', color = "red")

        # Show legend, which uses the label we defined for the plot above
        self.profit_ax.legend()
        legend = self.profit_ax.legend(loc=2, ncol=2, prop={'size': 8})
        legend.get_frame().set_alpha(0.4)

        last_date = date2num(self.df['Date'].values[current_step])
        last_profit = self.profit_values[current_step]

        # Annotate the current net worth on the net worth graph
        self.profit_ax.annotate('{0:.2f}'.format(profit), (last_date, last_profit),
                                   xytext=(last_date, last_profit),
                                   bbox=dict(boxstyle='round',
                                             fc='w', ec='k', lw=1),
                                   color="black",
                                   fontsize="small")
        # last_sp_profit = self.sp_profit[current_step]
        # self.profit_ax.annotate('{0:.2f}'.format(last_sp_profit), (last_date, last_sp_profit),
        #                            xytext=(last_date, last_sp_profit),
        #                            bbox=dict(boxstyle='round',
        #                                      fc='w', ec='k', lw=1),
        #                            color="black",
        #                            fontsize="small")
        min_profit = min(0, self.profit_values.min())
        max_profit = self.profit_values.max()
        # Add space above and below min/max net worth
        self.profit_ax.set_ylim(min_profit * 1.25, max_profit * 1.25)
            # min(self.profit_values[np.nonzero(self.profit_values)]) / 1.25, max(self.profit_values) * 1.25)


    def _render_price(self, current_step, net_worth, dates, step_range):
        self.price_ax.clear()

        # Format data for OHCL candlestick graph
        candlesticks = zip(dates,
                           self.df['Open'].values[step_range], self.df['Close'].values[step_range],
                           self.df['High'].values[step_range], self.df['Low'].values[step_range])

        # Plot price using candlestick graph from mpl_finance
        candlestick(self.price_ax, candlesticks, width=1,
                    colorup=UP_COLOR, colordown=DOWN_COLOR)

        last_date = date2num(self.df['Date'].values[current_step])
        last_close = self.df['Close'].values[current_step]
        last_high = self.df['High'].values[current_step]

        # Print the current price to the price axis
        self.price_ax.annotate('{0:.2f}'.format(last_close), (last_date, last_close),
                               xytext=(last_date, last_high),
                               bbox=dict(boxstyle='round',
                                         fc='w', ec='k', lw=1),
                               color="black",
                               fontsize="small")

        # Shift price axis up to give volume chart space
        ylim = self.price_ax.get_ylim()
        self.price_ax.set_ylim(ylim[0] - (ylim[1] - ylim[0])
                               * VOLUME_CHART_HEIGHT, ylim[1])

    def _render_volume(self, current_step, net_worth, dates, step_range):
        self.volume_ax.clear()

        volume = np.array(self.df['Volume'].values[step_range])

        pos = self.df['Open'].values[step_range] - \
            self.df['Close'].values[step_range] < 0
        neg = self.df['Open'].values[step_range] - \
            self.df['Close'].values[step_range] > 0

        # Color volume bars based on price direction on that date
        self.volume_ax.bar(dates[pos], volume[pos], color=UP_COLOR,
                           alpha=0.4, width=1, align='center')
        self.volume_ax.bar(dates[neg], volume[neg], color=DOWN_COLOR,
                           alpha=0.4, width=1, align='center')

        # Cap volume axis height below price chart and hide ticks
        self.volume_ax.set_ylim(0, max(volume) / VOLUME_CHART_HEIGHT)
        self.volume_ax.yaxis.set_ticks([])

    def _render_trades(self, current_step, trades, step_range):
        for trade in trades:
            if trade['step'] in step_range:
                date = date2num(self.df['Date'].values[trade['step']])
                high = self.df['High'].values[trade['step']]
                low = self.df['Low'].values[trade['step']]

                if trade['type'] == 'buy':
                    high_low = low
                    color = UP_TEXT_COLOR
                else:
                    high_low = high
                    color = DOWN_TEXT_COLOR

                total = '{0:.2f}'.format(trade['total'])

                # Print the current price to the price axis
                self.price_ax.annotate(f'${total}', (date, high_low),
                                       xytext=(date, high_low),
                                       color=color,
                                       fontsize=8,
                                       arrowprops=(dict(color=color)))

    def render(self, current_step, net_worth, trades, window_size=40):
        self.net_worths[current_step] = net_worth
        self.profit_values[current_step] = net_worth - 10000 # TODO: need to pass in a parameter in place of the hardcoded number

        window_start = max(current_step - window_size, 0)
        step_range = range(window_start, current_step + 1)

        # Format dates as timestamps, necessary for candlestick graph
        dates = np.array([date2num(x)
                          for x in self.df['Date'].values[step_range]])

        self._render_profit(current_step, net_worth, step_range, dates, 10000)

        self._render_net_worth(current_step, net_worth, step_range, dates)
        self._render_price(current_step, net_worth, dates, step_range)
        self._render_volume(current_step, net_worth, dates, step_range)
        self._render_trades(current_step, trades, step_range)

        # Format the date ticks to be more easily read
        self.price_ax.set_xticklabels(self.df['Date'].values[step_range], rotation=45,
                                      horizontalalignment='right')

        # Hide duplicate net worth date labels
        plt.setp(self.net_worth_ax.get_xticklabels(), visible=False)

        # Necessary to view frames before they are unrendered
        plt.pause(0.001)

    def close(self):
        plt.close()

#!/usr/bin/env python3

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List


#
# You are working on our new product that Product Management believes will allow
# us to take a significant amount of market share away from our our despised rival company
# Dodger-Baseball Inc.
#
# Our new product, called AutoGM, is revolutionary because it allows baseball general managers to quickly
# find the best player for a given baseball statistic.
#
# The data that the new product will use comes from a 3rd party API called BaseballNerds.
# The 3rd party API is in production, so we cannot change it's API structure, but they are a good partner so they
# are willing to accept constructive feedback on improvements.
# Unfortunately the BaseballNerds hosting provider is notoriously unreliable, though it almost always returns data eventually.
#
# You are tasked with the following user story:
# As a baseball general manager,
# I want a method to quickly find the best player with a given statistic.
# So that I can make in-game decisions in a timely manner.
#
# You are to:
# Design, Implement, and Test a function/method to use the 3rd Party API to return the necessary data.
#
# Note:
# * For simplicity all stats used are Counting stats, and Higher is Better.
# * In the interest of time the tests may be pseudo-code or simply descriptions of what would be tested.
# 	* Code commented list is fine; so is a main() method.
# * There is a simple log() method provided to just println what you would normally want to log
# * Ask as many clarifying questions, before, during or after as necessary.


###################################################################################################################
# Start of Baseball Nerds API SDK:
# NOTE: Normally this would exist in it's own package,
# but for simplicity of this test, it is just here and prefixed with BBN.
#
# !!Do not change these types!!
###################################################################################################################


@dataclass
class BBNPlayer:
    player_id: int
    """Unique player id."""
    first_name: str
    """Players First Name."""
    last_name: str
    """Players Last Name."""


@dataclass
class BBNBattingStatistic:
    player_id: int
    """Player this statistic belongs to."""
    rbi: int
    """Runs Batted In."""
    home_runs: int
    """Home Runs."""


@dataclass
class BBNPitchingStatistic:
    player_id: int
    """Player this statistic belongs to."""
    innings: int
    """Number of (whole) innings pitched."""
    strikeouts: int
    """Strikeouts."""


class BBNApiError(RuntimeError):
    """An Error occurred calling the BBN Api"""


class BBNPlayerApi(ABC):
    @abstractmethod
    def get_player(self, player_id: int) -> Optional[BBNPlayer]:
        """
        Get a player by PlayerID.
        :param player_id: player_id of the BBNPlayer to get.
        :return: BBNPlayer if id exists. Returns None if id does not exist.
        :raises: BBNApiError on API error.
        """

    @abstractmethod
    def list_players(self) -> List[BBNPlayer]:
        """
        List all Players.
        :return: List of BBNPlayers. Returns [] if no players exist.
        :raises: BBNApiError on API error.
        """


class BBNStatisticsApi(ABC):
    @abstractmethod
    def get_batting_stat(self, player_id: int) -> Optional[BBNBattingStatistic]:
        """
        Get a batting statistic by Player ID.
        :param player_id: player_id of the BBNPlayer to get batting statistic for.
        :return: BBNBattingStatistic if player_id and stat exists, else None.
        :raises: BBNApiError on API error.
        """

    @abstractmethod
    def get_pitching_stat(self, player_id: int) -> Optional[BBNPitchingStatistic]:
        """
        Get a pitching statistic by Player ID.
        :param player_id: player_id of the BBNPlayer to get batting statistic for.
        :return: BBNPitchingStatistic if player_id and stat exists, else None.
        :raises: BBNApiError on API error.
        """


class BBNTestClient(BBNPlayerApi, BBNStatisticsApi):
    def list_players(self) -> List[BBNPlayer]:
        players = [
            BBNPlayer(player_id=1, first_name="Mike", last_name="Trout"),
            BBNPlayer(player_id=3, first_name="Nolan", last_name="Arenado"),
            BBNPlayer(player_id=10, first_name="Kyle", last_name="Freeland"),
            BBNPlayer(player_id=11, first_name="Clayton", last_name="Kershaw"),
        ]
        return players

    def get_player(self, player_id: int) -> Optional[BBNPlayer]:
        players = self.list_players()
        for player in players:
            if player.player_id == player_id:
                return player
        return None

    def get_batting_stat(self, player_id: int) -> Optional[BBNBattingStatistic]:
        batting_stats = [
            BBNBattingStatistic(player_id=1, rbi=104, home_runs=45),
            BBNBattingStatistic(player_id=3, rbi=118, home_runs=41),
        ]
        for stat in batting_stats:
            if stat.player_id == player_id:
                return stat
        return None

    def get_pitching_stat(self, player_id: int) -> Optional[BBNPitchingStatistic]:
        pitching_stats = [
            BBNPitchingStatistic(player_id=10, innings=104, strikeouts=79),
            BBNPitchingStatistic(player_id=11, innings=178, strikeouts=189),
        ]
        for stat in pitching_stats:
            if stat.player_id == player_id:
                return stat
        return None


###################################################################################################################
# End of Baseball Nerds API SDK
###################################################################################################################


class APICalls(object):

    def gethighestStrikeoutsPlayers(self):
        bbnclient = BBNTestClient()
        data = bbnclient.list_players()
        highest = 0

        for item in data:
            # print(item.player_id)

            data2 = bbnclient.get_pitching_stat(item.player_id)

            if data2 is not None and data2.strikeouts > highest:
                highest = data2.strikeouts
        #print(highest)

        for playerName in data:
            print(playerName)

        # bbnclient.get_pitching_stat()


test = APICalls()
# bbnTests = BBNTestClient()
# print(bbnTests.get_pitching_stat(None))
print(test.getAllPlayers())

DEBUG = True




# You're code goes here:

if __name__ == '__main__':
    pass
